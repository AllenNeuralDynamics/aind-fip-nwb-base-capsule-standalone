"""
Tools for aligning FIP data to NWB time series
"""

import glob
import itertools
import json
import logging
import os
import re
import statistics
from datetime import datetime

import harp
import numpy as np
import pandas as pd
import pynwb
from hdmf_zarr import NWBZarrIO
from pynwb import NWBHDF5IO
from scipy import interpolate
from util.add_fiber_to_nwb import add_fiber_data_to_nwb

def is_numeric(obj):
    """
    Check if an array or object is numeric
    """
    attrs = ["__add__", "__sub__", "__mul__", "__truediv__", "__pow__"]
    return all(hasattr(obj, attr) for attr in attrs)


def split_fip_traces(df_fip, split_by=["channel", "fiber_number"]):
    """
    split_neural_traces takes in a dataframe with fiber photometry data series and splits it into
    individual traces for each channel and each channel number.

    Parameters
    ----------
    df_fip: DataFrame
        Time series Dataframe with columns signal, time, channel, and channel number.
        Has the signals for variations of channel and channel numbers are mixed together

    Returns
    ----------
    dict_fip: dictionary
        Dictionary that takes in channel name and channel number as key, and time series and signal
        bundled together as a 2x<TIMESERIES_LEN> as the value

    """
    dict_fip = {}
    groups = df_fip.groupby(split_by)
    for group_name, df_group in list(groups):
        df_group = df_group.sort_values("time")
        # Transforms integers in the name into int type strings. This is needed because nan in the dataframe entries automatically transform entire columns into float type
        group_name_string = [
            str(int(x)) if (is_numeric(x) and x == int(x)) else str(x)
            for x in group_name
        ]
        group_string = "_".join(group_name_string)
        dict_fip[group_string] = np.vstack(
            [df_group.time.values, df_group.signal.values]
        )
    return dict_fip


def num_rows(file_path):
    """Counts the number of rows in a file."""
    try:
        with open(file_path, "r") as file:
            return sum(1 for _ in file)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return 0


def detect_fiber_photometry_system(AnalDir):
    """
    Detects the fiber photometry system used by checking the presence of specific files in a directory.

    Parameters:
    ----------
    AnalDir : str
        The path to the directory where the analysis files are located.

    Returns:
    -------
    tuple
        A tuple containing:
        - system (str or None): The detected system type ("NPM" or "Homebrew"). Returns None if no system is detected.
        - filenames (list of str): A list of the relevant filenames for the detected system.
    """
    system = None
    filenames = []

    # Search for files matching each system's patterns
    for system_name, patterns in (
        ("NPM", ["L415", "L470", "L560"]),
        ("Homebrew", ["FIP_DataG", "FIP_DataR", "FIP_DataIso"]),
    ):
        for pattern in patterns:
            files = glob.glob(
                os.path.join(AnalDir, "**", f"{pattern}*"), recursive=True
            )
            if len(files) > 1:
                logging.error(
                    f"Multiple {pattern} recording files found: {files}."
                    " Using only the largest file."
                )
            if files:
                filenames.append(max(files, key=num_rows))  # Select the largest file
                system = system_name

    return system, filenames


def detect_behavior_recording_system(AnalDir):
    """
    This function detects the behavior recording system
    """
    AnalDir = AnalDir.as_posix()
    file_pavlovian = glob.glob(
        AnalDir + os.sep + "**" + os.sep + "TrialN_TrialType_ITI_*.csv", recursive=True
    )
    file_TTL = glob.glob(AnalDir + os.sep + "**" + os.sep + "TTL*")
    file_bonsai = glob.glob(
        AnalDir + os.sep + "FIP" + os.sep + "behavior" + os.sep + "*.json",
        recursive=True,
    )
    if file_bonsai == []:
        file_bonsai = glob.glob(AnalDir + os.sep + "behavior" + os.sep + "*.json")

    print(file_bonsai)
    behavior_system = None
    for file in file_bonsai:
        jsonFile = open(file, "r")
        # This try is needed because there exists json files which are not in json format (e.g.'/666613/FIP_666613_2023-05-12_17-25-27/notes.json')
        try:
            values = json.load(jsonFile)
        except:
            continue
        key_rising_time = [key for key in values.keys() if "Rising" in key]
        if len(key_rising_time):
            behavior_system = "bonsai"
            break

    if len(file_TTL):
        behavior_system = "bpod"

    # Note: this is not intended for long term use
    # We should remove the reliance on the behavior json
    # Soon
    if len(file_pavlovian):
        behavior_system = "pavlovian"
    return behavior_system


def load_NPM_fip_data(filenames, fibers_per_file=2):
    """
    This function loops over the filenames for the channels
    in the HomeBrew system DataIso, DataG, DataR

    Results
    The output dataframe has the following fields:
        - session
        - time
        - signal
        - fiber_number
        - channel
        - excitation
        - camera
        - system
    """

    df_fip = pd.DataFrame()
    for filename in filenames:

        # This assumes that the filename has a specific format
        subject_id, session_date, session_time = (
            re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", filename)
            .group()
            .split("_")
        )
        df_fip_file = pd.read_csv(filename)
        name = os.path.basename(filename)[:4]

        df_file = pd.DataFrame()

        # In the basic initial case the cameras are coupled to excitation and channel in a 1-1 correspondance
        camera = {"L415": "G", "L470": "G", "L560": "R"}[name]
        excitation = {"L415": 415, "L470": 470, "L560": 560}[name]
        channel = {"L415": "Iso", "L470": "G", "L560": "R"}[name]

        # Select the columns in the dataframe that correspond to the number of fibers
        columns = [
            col
            for col in df_fip_file.columns
            if ("Region" in col) & (camera == col[-1])
        ]
        columns = np.sort(columns)[:fibers_per_file]
        # Loop over the relevant columns for each camera/excitation
        for i_col, col in enumerate(columns):
            fiber_number = i_col
            df_fip_file_renamed = df_fip_file.loc[:, ["FrameCounter", "Timestamp", col]]
            df_fip_file_renamed = df_fip_file_renamed.rename(
                columns={
                    "FrameCounter": "frame_number",
                    "Timestamp": "time_fip",
                    col: "signal",
                }
            )
            df_fip_file_renamed.loc[:, "fiber_number"] = int(fiber_number)
            df_file = pd.concat([df_file, df_fip_file_renamed])

        # Assign fixed variables at the channel level to dataframe
        df_file.loc[:, "channel"] = channel
        df_file.loc[:, "excitation"] = excitation
        df_file.loc[:, "camera"] = camera

        # Concatenate dataframe
        df_fip = pd.concat([df_fip, df_file])

    # Assign fixed variables at the session level to dataframe
    df_fip.loc[:, "system"] = "NPM"
    df_fip.loc[:, "preprocess"] = "None"
    # Resort output dataframe
    df_fip_ses = df_fip.loc[
        :,
        [
            "frame_number",
            "time_fip",
            "signal",
            "channel",
            "fiber_number",
            "excitation",
            "camera",
            "system",
            "preprocess",
        ],
    ]
    return df_fip_ses


def clean_timestamps_NPM(df_fip_ses):
    """
    This function cleans up timestamps in a DataFrame to verify that the frequency of the entry of the timestamps is fixed.
    It adds missing timestamps where necessary.

    The function automatically converts milliseconds to seconds if differences between timestamps are > 10.
    """
    # If the DataFrame is empty, return it as is
    if not len(df_fip_ses):
        return df_fip_ses

    # Get the unique channels in the DataFrame
    channels = pd.unique(df_fip_ses["channel"])

    # Initialize an empty DataFrame
    df_fip_ses_cleaned = pd.DataFrame()

    # Loop over each channel
    for i_channel, channel in enumerate(channels):
        # Filter the DataFrame for the current channel
        df_iter_channel = df_fip_ses[df_fip_ses["channel"] == channel]

        # Get the unique channel numbers for the current channel
        fiber_numbers = pd.unique(df_iter_channel["fiber_number"])

        # Loop over each channel number
        for fiber_number in fiber_numbers:
            # Filter the DataFrame for the current channel number
            df_iter = df_iter_channel[df_iter_channel["fiber_number"] == fiber_number]

            # If the DataFrame is empty, skip to the next iteration
            if not len(df_iter):
                continue

            # If the maximum difference between timestamps is greater than 10, convert the timestamps to seconds
            if np.max(np.diff(df_iter["time_fip"])) > 10:
                df_iter.loc[:, "time_fip"] = df_iter["time_fip"] / 1000.0

            # Get the timestamps for the current DataFrame
            timestamps_fip = df_iter["time_fip"].values

            # Calculate the mode and frequency of the differences between timestamps
            mode_fip = statistics.mode(np.round(np.diff(timestamps_fip), 2))
            frequency = int(1 / mode_fip)

            # If the frequency is not 20, print a message
            if frequency != 20:
                print(
                    "    "
                    + channel
                    + " The frequencies of the timestamps are FIP "
                    + str(frequency)
                )

            # Clean the timestamps and Merge the cleaned timestamps with the current DataFrame
            timestamps_fip_cleaned = clean_timestamps(timestamps_fip, pace=mode_fip)
            df_iter_cleaned = pd.merge(
                pd.DataFrame(timestamps_fip_cleaned, columns=["time_fip"]),
                df_iter,
                on="time_fip",
                how="left",
            )
            df_fip_ses_cleaned = pd.concat(
                [df_fip_ses_cleaned, df_iter_cleaned], axis=0
            )

    # Return the cleaned DataFrame
    return df_fip_ses_cleaned


def alignment_fip_time_to_bpod(AnalDir, folder_nwb="/data/foraging_nwb_bpod/"):
    """
    This computes a function that transforms frame numbers of the fiber photometry camera into timestamps of the behavior recording system NPM
    Such alignment between NPM and Bpod is based on the bitcode system developed by Han Hao (ask Han or Kenta Hagihara for more details).
    The function takes the bitcodes in both systems and returns and interpolator for the respective timepoints.
        - AnalDir is the folder of fiber photometry data
        - folder_nwb is the folder of bpod data

    Output
        - frame_number_convertor_FIP_to_NWB a function
    """

    # Get the name of the session and retrieve the corresponding nwb file
    subject_id, session_date, session_time = (
        re.search(r"\d{6}_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", AnalDir)
        .group()
        .split("_")
    )
    session_name = f"{subject_id}_{session_date}_"
    filename_nwb = glob.glob(folder_nwb + session_name + "*.nwb")
    if len(filename_nwb) == 0:
        print(
            "Could not process session because of missing nwb (e.g. session with no behavior)"
        )
        return np.nan
    io = NWBHDF5IO(filename_nwb[0], mode="r")
    nwb = io.read()

    # Read bitcodes and go cue times from the BPOD NWB file
    df_trials = nwb.trials.to_dataframe()
    timestamps_bitcodes_NWB, bitcodes_NWB = (
        df_trials.goCue_start_time.values,
        df_trials.bpod_backup_trial_bit_code.values,
    )

    # Compute the bitcodes, go cue time and corresponding frame numbers from the fiber photometry NPM system files
    (
        timestamps_bitcodes_FIP,
        frame_number_bitcodes_FIP,
        bitcodes_FIP,
    ) = compute_timestamps_bitcodes(AnalDir)
    if (len(frame_number_bitcodes_FIP) != len(bitcodes_FIP)) or len(bitcodes_FIP) == 1:
        return np.nan
    # Retain only bitcodes that are reported in both systems
    bitcodes, idxs_FIP, idxs_NWB = np.intersect1d(
        bitcodes_FIP, bitcodes_NWB, return_indices=True
    )
    if len(bitcodes) == 0:
        print("Bitcodes between system could not be matched")
        return np.nan

    idxs = np.argsort(idxs_NWB)
    bitcodes, idxs_FIPs, idxs_NWBs = bitcodes[idxs], idxs_FIP[idxs], idxs_NWB[idxs]

    # Create interpolating functions from timestamps and frame_numbers
    # timestamps_convertor_FIP_to_NWB = interpolate.interp1d(timestamps_bitcodes_FIP[idxs_FIPs], timestamps_bitcodes_NWB[idxs_NWBs], fill_value='extrapolate')
    frame_number_convertor_FIP_to_NWB = interpolate.interp1d(
        frame_number_bitcodes_FIP[idxs_FIPs],
        timestamps_bitcodes_NWB[idxs_NWBs],
        fill_value="extrapolate",
    )

    return frame_number_convertor_FIP_to_NWB


def compute_timestamps_bitcodes(AnalDir):
    """
    This function extracts bitcodes and corresponding times or frame numbers from the TTL stored files.

    Output
        - timestamps_bitcodes
        - frame_number_bitcodes
        - bitcodes
    """

    TTL_signal = np.fromfile(glob.glob(AnalDir + os.sep + "**" + os.sep + "TTL_20*")[0])
    file_TTLTS = glob.glob(AnalDir + os.sep + "**" + os.sep + "TTL_TS*")[0]
    TTL_ts = pd.read_csv(file_TTLTS, header=None).values.flatten()

    # % Sorting NIDAQ-AI channels
    if (len(TTL_signal) / 1000) / len(TTL_ts) == 1:
        time_seconds = np.arange(len(TTL_signal)) / 1000
        frame_number = np.arange(len(TTL_signal))

    elif (len(TTL_signal) / 1000) / len(
        TTL_ts
    ) == 2:  # this shouldn't happen, though...
        TTL_signal_ini = TTL_signal[0::2]
        time_seconds = np.arange(len(TTL_signal_ini)) / 1000
        frame_number = np.arange(len(TTL_signal))[::2]

    elif (len(TTL_signal) / 1000) / len(TTL_ts) >= 3:
        TTL_signal_ini = TTL_signal[::3]
        time_seconds = np.arange(len(TTL_signal_ini)) / 1000
        frame_number = np.arange(len(TTL_signal))[::3]
    else:
        print("Something is wrong with TimeStamps or Analog Recording...")
        return [np.nan], [np.nan], [np.nan]

    # % analoginputs binalize
    TTL_signal_thresholded = (TTL_signal_ini > 3).astype(int)
    diff = np.diff(TTL_signal_thresholded, prepend=0)

    # Find indices where diff is 1
    indices_1 = np.where(diff == 1)[0]

    # For each index, find the next index where diff is -1 within a range of 120
    # next_indices = [np.where(diff[ii:ii+120] == -1)[0][0] for ii in indices]

    # This is the robust replacement of the one-liner above
    indices, next_indices = [], []
    for index in indices_1:
        next_index_list = np.where(diff[index : index + 120] == -1)[0]
        if len(next_index_list):
            indices += [index]
            next_indices += [next_index_list[0]]

    if len(indices) == 0:
        print("Something is wrong with Analog Recording...")
        return [np.nan], [np.nan], [np.nan]

    # Convert lists to numpy arrays
    TTL_position = np.array(indices)
    TTL_length = np.array(next_indices)
    time_seconds_TTL_position = time_seconds[indices]
    frame_number_position = frame_number[indices]

    # This cleans up the TTL signal into predetermined values
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0])
    bin_edges = np.concatenate([values[:-1] + np.diff(values) / 2, [values[-1] + 10]])
    idxs = np.digitize(TTL_length, bin_edges)
    TTL_length = values[idxs]

    # Filter indices where TTL_l is 20
    indices_barcode = np.where(TTL_length == 20)[
        0
    ]  # This corresponds to times where the barcode is passed
    indices_times = np.where(TTL_length == 1)[0]  # This corresponds to gocues

    if len(indices_times) > len(indices_barcode):
        indices_barcodes_keep = (
            np.argmax(
                (
                    indices_times[:, np.newaxis]
                    - indices_barcode[:, np.newaxis].transpose()
                )
                < 0,
                axis=1,
            )
            - 1
        )
        indices_barcode = indices_barcode[indices_barcodes_keep]
    if len(indices_times) < len(indices_barcode):
        indices_times_keep = (
            np.argmax(
                (
                    indices_barcode[:, np.newaxis]
                    - indices_times[:, np.newaxis].transpose()
                )
                < 0,
                axis=1,
            )
            - 1
        )
        indices_times = indices_barcode[indices_times_keep]

    # Extract relevant data
    BarcodeP = TTL_position[indices_barcode]
    time_seconds_BarcodeP = time_seconds_TTL_position[indices_times]
    frame_number_BarcodeP = frame_number_position[indices_times]
    # BarcodeP_1k = TTL_p_align_1k[indices_barcode]

    # Calculate the indices for TTLsignal1
    offsets = np.arange(20) * 20 + 30 + 5
    indices_barcode_vectorial = BarcodeP[:, np.newaxis] + offsets

    # Vectorized operation to populate BarcodeBin
    BarcodeBin = TTL_signal_ini[indices_barcode_vectorial]

    # Convert BarcodeBin to BarChar
    bitcodes = np.array(
        ["".join(map(str, (row > 3).astype(int))) for row in BarcodeBin]
    )
    # timestamps_bitcodes = BarcodeP_1k
    timestamps_bitcodes = time_seconds_BarcodeP
    frame_number_bitcodes = frame_number_BarcodeP
    # add bitcodes storing
    return timestamps_bitcodes, frame_number_bitcodes, bitcodes


def load_Homebrew_fip_data(filenames, fibers_per_file=2):
    """
    This function loops over the filenames for the channels
    in the NPM system 'L415', 'L470', 'L560'
    The created dataframe has the following fields:
        - session
        - time
        - signal
        - fiber_number
        - channel
        - excitation
        - camera
        - system
    """

    df_fip = pd.DataFrame()
    save_fip_channels = np.arange(1, fibers_per_file + 1)
    for filename in filenames:
        header = os.path.basename(filename).split("/")[-1]
        channel = ("_".join(header.split("_")[:2])).replace("FIP_Data", "")
        if "Raw" in channel:
            continue
        try:
            if ".csv" in filename:
                df_fip_file = pd.read_csv(filename, header=None)  # read the CSV file
                zero_columns = (df_fip_file == 0).all(axis=0)
                if zero_columns.any():
                    logging.warning(
                        f"CSV file '{filename}' contains {zero_columns.sum()} column(s) with all zeros. "
                        f"Removing these columns to address a bug in older FIP_DataIso.csv files."
                    )
                    df_fip_file = df_fip_file.loc[:, ~zero_columns]
                    df_fip_file.columns = range(len(df_fip_file.columns))
        except pd.errors.EmptyDataError:
            continue
        except FileNotFoundError:
            continue
        df_file = pd.DataFrame()
        for col in df_fip_file.columns[save_fip_channels]:
            df_fip_file_renamed = df_fip_file[[0, col]].rename(
                columns={0: "time_fip", col: "signal"}
            )
            channel_number = int(col)
            df_fip_file_renamed["fiber_number"] = channel_number - 1
            df_fip_file_renamed.loc[:, "frame_number"] = df_fip_file.index.values
            df_file = pd.concat([df_file, df_fip_file_renamed])
        df_file["channel"] = channel
        camera = {"Iso": "G", "G": "G", "R": "R"}[channel]
        excitation = {"Iso": 415, "G": 470, "R": 560}[channel]
        df_file["excitation"] = excitation
        df_file["camera"] = camera
        df_fip = pd.concat([df_fip, df_file], axis=0)

    if len(df_fip) > 0:
        df_fip["system"] = "FIP"
        df_fip["preprocess"] = "None"
        df_fip_ses = df_fip.loc[
            :,
            [
                "frame_number",
                "time_fip",
                "signal",
                "channel",
                "fiber_number",
                "excitation",
                "camera",
                "system",
                "preprocess",
            ],
        ]
    else:
        df_fip_ses = df_fip
    return df_fip_ses


def to_daytime(t="2024-09-13 09:47:02.548400"):
    if isinstance(t, str):
        try:
            d = datetime.fromisoformat(t)
            return d.hour * 3600 + d.minute * 60 + d.second + d.microsecond / 1000000
        except:
            return t
    else:
        return t


def clean_timestamps_Harp(AnalDir, timestamps_fip, diff_counts, max_drop):
    """
    Cleans Harp timestamps from raw.harp/BehaviorEvents/Event_32.bin in the specified directory.

    This function processes timestamp data from raw.harp/BehaviorEvents/Event_32.bin located in the
    given directory. It ensures the timestamps are cleaned and aligned for further analysis. If the
    data is incomplete, corrupted, or contains large gaps, appropriate errors or warnings are raised.

    Args:
        AnalDir (str): The directory containing the raw.harp directory with timestamp data.
        timestamps_fip (np.ndarray): FIP timestamps from longest (most rows) CSV file.
        diff_counts (int): Difference in number of FIP timestamps between channels.
        max_drop (int): The maximum number of timestamps that can be dropped from the
                        start and end of the dataset during truncation.

    Returns:
        np.ndarray: A NumPy array of cleaned and aligned Harp timestamps.

    Raises:
        ValueError: If the timestamps cannot be successfully cleaned and aligned.

    Notes:
        - If a large gap is detected in the timestamps, the function attempts to truncate
          the data up to the point where the gap appears if it happens within the first or
          last max_drop frames. If the gap occurs mid session, it is kept.
    """
    data_raw = harp.read(f"{AnalDir}/behavior/raw.harp/BehaviorEvents/Event_32.bin")
    data = data_raw[0].values
    asint = (data & 0x04 > 0).astype(int)
    rising_edges = np.where(np.diff(asint, prepend=0) == 1)[0]
    timestamps_Harp = data_raw.index[rising_edges].values
    T = len(timestamps_Harp)
    dt_fip = np.diff(timestamps_fip) / 1000
    dt_harp = np.diff(timestamps_Harp)
    neg = np.where(dt_harp < 0)[0]
    if len(neg):
        for i in neg:
            if i + 2 < T:  # interpolate
                timestamps_Harp[i + 1] = (
                    timestamps_Harp[i] + timestamps_Harp[i + 2]
                ) / 2
            else:  # extrapolate
                timestamps_Harp[i + 1] = 2 * timestamps_Harp[i] - timestamps_Harp[i - 1]
        dt_harp = np.diff(timestamps_Harp)
    gap_times = np.where(dt_harp > 0.2)[0]
    if len(gap_times):
        for i in gap_times:
            logging.warning(
                f"Large gap in HARP timestamps between index {i} and {i+1} out of"
                f" {T} of size {dt_harp[i]:.4f}s"
            )
        segment_times = np.hstack([[0], gap_times + 1, T])
        seg_start = np.argmax(np.diff(segment_times))
        seg_end = seg_start + 1
        longest_segment = segment_times[seg_start : seg_end + 1]
        logging.warning(
            f"Longest contiguous block of regular HARP timestamps: "
            f"[{longest_segment[0]}, {longest_segment[1]})."
        )
        near_start = longest_segment[0] <= max_drop
        near_end = longest_segment[1] >= T - max_drop
        kept_gaps = []
        while ~near_start or ~near_end:
            # combine segments tolerating gaps until the longest_segment
            # starts before (<=) max_drop and ends after (>=) T-max_drop
            i = longest_segment[0 if ~near_start else 1]
            logging.error(
                f"Gap in HARP timestamps between index {i-1} and {i} out of "
                f"{T} of size {dt_harp[i-1]:.4f}s occurs mid-session, "
                "hence keeping the gap!"
            )
            kept_gaps.append((int(i - 1), float(dt_harp[i - 1])))
            if ~near_start:
                seg_start -= 1
            else:
                seg_end += 1
            longest_segment = [segment_times[seg_start], segment_times[seg_end]]
            near_start = longest_segment[0] <= max_drop
            near_end = longest_segment[1] >= T - max_drop
        gap_times_fip = np.where(dt_fip > 0.2)[0]
        if len(gap_times_fip) == 0:
            drop = longest_segment[0]
        else:
            drop_candidates = [
                (x - y) for x, y in itertools.product(gap_times, gap_times_fip)
            ]
            drop_candidates = np.array(drop_candidates)[
                np.abs(drop_candidates) < T // 2
            ]
            Tmin = min(T, len(timestamps_fip)) - 1
            cc = [
                np.dot(
                    dt_harp[max(0, d) : min(Tmin, Tmin + d)],
                    dt_fip[max(0, -d) : min(Tmin, Tmin - d)],
                )
                / (Tmin - abs(d))
                for d in drop_candidates
            ]
            drop = drop_candidates[np.argmax(cc)]
            corr = np.corrcoef(
                dt_harp[max(0, drop) : min(Tmin, Tmin + drop)],
                dt_fip[max(0, -drop) : min(Tmin, Tmin - drop)],
            )[0, 1]
            matches_gaps = np.intersect1d(gap_times, gap_times_fip + drop)
            logging.warning(
                "Matching gaps in HARP and FIP timestamps suggests dropping the "
                f"first {abs(drop)} {'HARP' if drop >=0 else 'FIP'} timestamps. "
                f"This achieves a cross-correlation of {np.max(cc):.4f} and Pearson "
                f"correlation of {corr:.4f} for the timestamp intervals."
            )
            for i_harp in matches_gaps:
                i_fip = i_harp - drop
                logging.warning(
                    f"This matches gap in HARP timestamps between index {i_harp} "
                    f"and {i_harp+1} out of {T} of size {dt_harp[i_harp]:.4f}s\n"
                    "with mirrored gap in FIP timestamps between index "
                    f"{i_fip} and {i_fip+1} out of {len(timestamps_fip)} "
                    f"of size {dt_fip[i_fip]:.4f}s."
                )
        drop_fip = len(timestamps_fip) - longest_segment[1] + drop
        drop_start = int(max(drop, longest_segment[0], diff_counts))
        drop_end = int(T - longest_segment[1] - min(0, drop_fip))
        logging.warning(
            f"Dropping {drop_start} HARP timestamps at start and {drop_end} at end.\n"
            f"Dropping {max(0, longest_segment[0]-drop)} FIP timestamps at "
            f"start and {max(0, drop_fip)} at end."
        )
        # Drop timestamps in HARP and FIP by setting them to nan, see `deal_with_nans`
        timestamps_Harp_cleaned = np.full_like(timestamps_Harp, np.nan)
        timestamps_Harp_cleaned[slice(*longest_segment)] = timestamps_Harp[
            slice(*longest_segment)
        ]
        return timestamps_Harp_cleaned[drop:], drop_start, drop_end, kept_gaps
    else:
        drop_end = max(0, T - len(timestamps_fip))
        if drop_end:
            # we usually align at the start, the following checks whether to rather align at the end
            start, end = np.nan, np.nan
            for file in [
                f for f in glob.glob(f"{AnalDir}/behavior/*.json") if "model" not in f
            ]:
                with open(file, "r") as jsonFile:
                    values = json.load(jsonFile)
                start = timestamps_fip[0] / 1000 - to_daytime(
                    values.get("fiber_photometry_start_time", np.nan)
                )
                end = timestamps_fip[-1] / 1000 - to_daytime(
                    values.get("fiber_photometry_end_time", np.nan)
                )
            if (start < 0.9 or start > 1.2) and end >= 0 and end <= 1.2:
                logging.warning(
                    f"Unusual time difference: first CSV timestamp vs. "
                    "JSON `fiber_photometry_start_time`, but\n"
                    f"typical time difference:  last CSV timestamp vs. "
                    "JSON `fiber_photometry_end_time`,\n"
                    "thus aligning at the end rather than the start."
                )
                logging.warning(
                    f"Dropping {diff_counts + drop_end} HARP timestamps at start and 0 at end."
                )
                return timestamps_Harp[drop_end:], diff_counts + drop_end, 0, []
        return timestamps_Harp, diff_counts, drop_end, []
        if drop_end or diff_counts:
            logging.warning(
                f"Dropping {diff_counts} HARP timestamps at start and {drop_end} at end."
            )
        return timestamps_Harp, diff_counts, drop_end, []


def alignment_fip_time_to_harp(
    df_fip_ses, timestamps_Harp_cleaned, split_by=["channel", "fiber_number"]
):
    if not len(df_fip_ses):
        return df_fip_ses

    channel_number = pd.unique(df_fip_ses["fiber_number"])[0]
    channels = pd.unique(df_fip_ses["channel"])
    channels = channels[~pd.isna(channels)]
    df_fip_sel = df_fip_ses[
        (df_fip_ses["preprocess"] == "None")
        & (df_fip_ses["fiber_number"] == channel_number)
    ]

    timestamps = {}
    for channel in channels:
        timestamps_channel = df_fip_sel[df_fip_sel["channel"] == channel][
            "time_fip"
        ].values
        conversion_factor = np.round(
            statistics.mode(np.round(np.diff(timestamps_channel), 2)) / 0.05
        )
        power = 10 ** round(np.log10(abs(conversion_factor)))
        timestamps[channel] = timestamps_channel / power

    if len(nt := np.unique([len(timestamps[channel]) for channel in channels])) == 1:
        logging.info(f"All channels contain the same number of timestamps: {nt}.")
        reference_channel = "G"
    else:
        logging.warning(f"Channels contain different number of timestamps: {nt}.")
        idx = np.argmax([len(timestamps[key]) for key in timestamps.keys()])
        reference_channel = list(timestamps.keys())[idx]

    timestamps_ref = timestamps[reference_channel]
    logging.info(f"Number of cleaned harp timestamps: {len(timestamps_Harp_cleaned)}")
    if len(timestamps_Harp_cleaned) > len(timestamps_ref):
        timestamps_Harp_padded = timestamps_Harp_cleaned[: len(timestamps_ref)]
    else:
        timestamps_Harp_padded = np.concatenate(
            [
                timestamps_Harp_cleaned,
                [np.nan] * (len(timestamps_ref) - len(timestamps_Harp_cleaned)),
            ],
        )

    timestamps_new = {}
    for channel in channels:
        sub_idxs = np.linspace(
            0, len(timestamps_Harp_padded) - 1, len(timestamps[channel])
        ).astype(int)
        time_shifts = timestamps[channel] - timestamps_ref[sub_idxs]
        timestamps_new[channel] = timestamps_Harp_padded[sub_idxs] + time_shifts

    df_fip_ses_aligned = pd.DataFrame()
    groups = df_fip_ses.groupby(split_by)
    for group_name, df_group in groups:
        channel_idx = np.where(np.array(split_by) == "channel")[0][0]
        df_group.loc[:, "time"] = timestamps_new[group_name[channel_idx]]
        df_group.loc[:, "time_fip"] = np.round(df_group["time_fip"].values, 4)
        df_fip_ses_aligned = pd.concat([df_fip_ses_aligned, df_group])

    return df_fip_ses_aligned


def clean_timestamps(timestamps, pace=0.05, tolerance=0.2):
    """
    This function cleans up a list of timestamps by removing and filling in values based on a given pace and tolerance.

    Parameters:
    timestamps (numpy array): The original timestamps to be cleaned.
    pace (float): The expected difference between consecutive timestamps.
    tolerance (float): The percentage of the pace that is acceptable as a deviation.

    Returns:
    timestamps_final (numpy array): The cleaned timestamps.
    """

    # Calculate the thresholds for removal and filling in of timestamps
    threshold_remove = pace - pace * tolerance
    threshold_fillin = pace + pace * tolerance
    # Identify the timestamps to keep based on the removal threshold
    idxs_to_keep = np.diff(timestamps, prepend=-np.inf) >= threshold_remove
    timestamps_cleaned = timestamps[idxs_to_keep]
    # Initialize the final timestamps
    timestamps_final = timestamps_cleaned
    # Loop until all gaps larger than the fill-in threshold are filled
    while any(np.diff(timestamps_final) > threshold_fillin):
        # Identify the first gap to fill
        idx_to_fillin = np.where(np.diff(timestamps_final) > threshold_fillin)[0][0]
        # Calculate the size of the gap
        gap_to_fillin = np.round(np.diff(timestamps_final)[idx_to_fillin], 2)
        # Generate the values to fill in the gap
        values_to_fillin = timestamps_final[idx_to_fillin] + np.arange(
            pace, gap_to_fillin, pace
        )
        # Insert the new values into the final timestamps
        timestamps_final = np.insert(
            timestamps_final, idx_to_fillin + 1, values_to_fillin
        )
    return timestamps_final


def append_aligned_fiber_to_nwb(AnalDir, max_drop, nwb_file):
    """Combine preprocessing steps and append to nwb"""
    drop_start, drop_end, kept_gaps = 0, 0, []
    # % Detect Fiber photometry and behavioral systems
    fiber_photometry_system, filenames = detect_fiber_photometry_system(AnalDir)
    behavior_system = detect_behavior_recording_system(AnalDir)

    if fiber_photometry_system is None or behavior_system is None:
        print(fiber_photometry_system, behavior_system)
        raise ValueError("FIP or behavior system not detect")

    # % Load FIP data and create fip dataframe
    if fiber_photometry_system == "NPM":
        df_fip_ses = load_NPM_fip_data(filenames, fibers_per_file=fibers_per_file)
        df_fip_ses_cleaned = clean_timestamps_NPM(df_fip_ses)

    elif fiber_photometry_system == "Homebrew":
        df_fip_ses_cleaned = load_Homebrew_fip_data(filenames, fibers_per_file=4)

    if len(df_fip_ses_cleaned) == 0:
        print("Could not processs session because of loading the data:" + session_name)
        raise ValueError(
            "Could not processs session because of loading the data:" + session_name
        )

    # % Align FIP to behavioral system clock
    if behavior_system == "bpod":
        (
            timestamps_bitcodes,
            frame_number_bitcodes,
            bitcodes,
        ) = compute_timestamps_bitcodes(AnalDir)
        frame_number_convertor_FIP_to_bpod = alignment_fip_time_to_bpod(
            AnalDir, folder_nwb="/data/foraging_nwb_bpod/"
        )
        if frame_number_convertor_FIP_to_bpod is np.nan:
            print(
                "Could not processs session because of bitcodes alignment:"
                + session_name
            )
            raise ValueError(
                "Could not processs session because of bitcodes alignment:"
                + session_name
            )

        df_fip_ses_cleaned.loc[:, "time"] = frame_number_convertor_FIP_to_bpod(
            df_fip_ses_cleaned["frame_number"].values
        )
        df_fip_ses_aligned = df_fip_ses_cleaned.loc[
            :,
            [
                "session",
                "frame_number",
                "time",
                "time_fip",
                "signal",
                "channel",
                "fiber_number",
                "excitation",
                "camera",
                "system",
            ],
        ]

    elif behavior_system == "bonsai":
        df0 = df_fip_ses_cleaned[["time_fip", "channel"]][
            df_fip_ses_cleaned["fiber_number"] == 0
        ]
        counts = df0["channel"].value_counts()
        diff_counts = max(counts) - min(counts)
        if diff_counts:
            for ch in (counts.idxmax(), counts.idxmin()):
                with pd.option_context("display.float_format", "{:.3f}".format):
                    logging.info(df0[df0["channel"] == ch].iloc[[0, 1, 2, 3, -1]])
        timestamps_fip = df0["time_fip"][df0["channel"] == counts.idxmax()].values
        timestamps_Harp_cleaned, drop_start, drop_end, kept_gaps = (
            clean_timestamps_Harp(AnalDir, timestamps_fip, diff_counts, max_drop)
        )
        df_fip_ses_aligned = alignment_fip_time_to_harp(
            df_fip_ses_cleaned, timestamps_Harp_cleaned
        )

    # Pavlovian has no times to align with currently
    if behavior_system == "pavlovian":
        # Note: we have no 'time' because it is not aligned
        # Use the raw fip times for now
        df_fip_ses_cleaned["time"] = df_fip_ses_cleaned["time_fip"]
        dict_fip = split_fip_traces(
            df_fip_ses_cleaned, split_by=["channel", "fiber_number"]
        )

    # Other types can be aligned with behavior
    else:
        dict_fip = split_fip_traces(
            df_fip_ses_aligned, split_by=["channel", "fiber_number"]
        )

    dict_fip = deal_with_nans(dict_fip, check_both=True, max_drop=max_drop)
    nwbfile = add_fiber_data_to_nwb(nwb_file, dict_fip=dict_fip)
    return nwbfile, drop_start, drop_end, kept_gaps, behavior_system


def attach_dict_fip(AnalDir, folder_nwb, dict_fip):
    filename_nwb = glob.glob(folder_nwb + "*.nwb")
    src_io = NWBZarrIO(filename_nwb[0], mode="r+")
    nwb = src_io.read()
    for neural_stream in dict_fip:
        ts = pynwb.TimeSeries(
            name=neural_stream,
            data=dict_fip[neural_stream][1],
            unit="s",
            timestamps=dict_fip[neural_stream][0],
        )
        nwb.add_acquisition(ts)
    return nwb


def first_nan_idx(arr, axis=-1):
    """
    Find the index of the first NaN along the specified axis.

    Parameters:
    arr (np.ndarray): Input array.
    axis (int): Axis along which to find the first NaN.

    Returns:
    int: Index of the first NaN along the specified axis, or -1 if no NaN is found.
    """
    is_nan = np.isnan(arr)
    if ~np.any(is_nan):
        return -1
    first_nan_idx = np.argmax(is_nan, axis=axis)
    return min(first_nan_idx[np.any(is_nan, axis=-1)])


def first_nonan_idx(arr, axis=-1):
    """
    Find the index of the first non-NaN along the specified axis.

    Parameters:
    arr (np.ndarray): Input array.
    axis (int): Axis along which to find the first non-NaN.

    Returns:
    int: Index of the first non-NaN along the specified axis, or -1 if only NaNs are found.
    """
    is_nan = np.isnan(arr)
    if ~np.any(~is_nan):
        return -1
    first_nonan_idx = np.argmax(~is_nan, axis=axis)
    return max(first_nonan_idx[np.any(~is_nan, axis=-1)])


def deal_with_nans(dict_fip, check_both=True, max_drop=12000):
    """
    Handle NaNs and length differences in a dictionary of data traces.

    Each array in the dictionary has the shape (2, T) and contains timestamps in the first row
    and data in the second row. This function identifies the first occurrence of NaN in each trace
    and adjusts the length of the traces accordingly.

    Parameters:
    dict_fip (dict): Dictionary with keys as identifiers and values as 2D numpy arrays (2, T).
    check_both (bool): If True, check both rows for NaNs; otherwise, check only the second row (data).
    max_drop (int): Maximum number of samples to drop from the end if NaNs are found near the end.

    Returns:
    dict: The input dictionary with traces truncated to remove NaNs.

    Notes:
    - If no NaNs are found, the traces are truncated to the minimum length among all traces.
    - If NaNs are found near the start of the traces (by default within the first 12000 samples,
      i.e. 10mins x 20Hz), the traces are truncated to start at the first non-NaN occurrence.
    - If NaNs are found near the end of the traces (by default within the last 12000 samples,
      i.e. 10mins x 20Hz), the traces are truncated to the length of the first NaN occurrence.
    - If NaNs are found in the middle of the traces, a warning is issued and
      the traces are truncated to the minimum length among all traces.
    """
    ks = list(dict_fip.keys())
    n_frames = [dict_fip[k].shape[1] for k in ks]
    min_len, max_len = min(n_frames), max(n_frames)
    min_ch, max_ch = ks[np.argmin(n_frames)], ks[np.argmax(n_frames)]
    if min_len < max_len:
        min_first, min_last = (dict_fip[min_ch][0, i] for i in (0, -1))
        max_first, max_last = (dict_fip[max_ch][0, i] for i in (0, -1))
        logging.warning(
            f"Warning! Shortest/longest trace has {min_len}/{max_len} frames.\n"
            "Dropping initial frames: fiber CSV aligns at end.\n"
            f"Shortest trace {min_ch:5} has first/last timestamp {min_first:.3f}s/{min_last:.3f}s.\n"
            f" Longest trace {max_ch:5} has first/last timestamp {max_first:.3f}s/{max_last:.3f}s.\n"
            f"The first/last timestamps differ by {min_first - max_first:.3f}s/{min_last - max_last:.3f}s"
        )
        for k in dict_fip.keys():
            dict_fip[k] = dict_fip[k][:, -min_len:]

    first_nonan = np.array(
        [
            first_nonan_idx(dict_fip[k][slice(None) if check_both else 1])
            for k in dict_fip.keys()
        ]
    )
    first_nonan = max(first_nonan[first_nonan > -1])
    first_nan = np.array(
        [
            first_nan_idx(dict_fip[k][slice(None) if check_both else 1, first_nonan:])
            for k in dict_fip.keys()
        ]
    )
    first_nan[first_nan > -1] += first_nonan
    first_nan = -1 if np.all(first_nan == -1) else min(first_nan[first_nan > -1])

    if first_nan == -1:
        new_len = min_len
    elif first_nan > min_len - max_drop:
        new_len = min(min_len, first_nan)
        logging.warning(
            f"Trace includes NaN near the end. Dropping last {min_len - new_len} frames."
        )
    else:
        new_len = min_len
        logging.warning("Warning! Trace includes NaN in the middle.")

    if first_nonan > max_drop or first_nonan == -1:
        first_nonan = 0
        logging.warning("Warning! Trace includes NaN in the middle.")

    elif first_nonan > 0:
        logging.warning(
            f"Trace includes NaN near the start. Dropping first {first_nonan} frames."
        )

    for k in dict_fip.keys():
        dict_fip[k] = dict_fip[k][:, first_nonan:new_len]

    return dict_fip
