import glob
import json
import logging
import os
from datetime import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aind_data_schema.core.quality_control import (
    QCEvaluation,
    QCMetric,
    QCStatus,
    QualityControl,
    Stage,
    Status,
)
from aind_data_schema_models.modalities import Modality
from aind_qcportal_schema.metric_value import DropdownMetric
from util.aligned_fiber_to_nwb_legacy import num_rows, to_daytime
from harp import read
from hdmf_zarr import NWBZarrIO
from scipy.stats import ttest_1samp


def get_GoCueTime(fiber_path):
    for file in [
        f for f in glob.glob(f"{fiber_path}/behavior/*.json") if "model" not in f
    ]:
        with open(file, "r") as jsonFile:
            values = json.load(jsonFile)
    return values["B_GoCueTime"]


def timestamps_from_harpraw(fiber_path):
    data_raw = read(f"{fiber_path}/behavior/raw.harp/BehaviorEvents/Event_32.bin")
    data = data_raw[0].values
    asint = (data & 0x04 > 0).astype(int)
    rising_edges = np.where(np.diff(asint, prepend=0) == 1)[0]
    timestamps_raw = data_raw.index[rising_edges]
    return timestamps_raw


def dataframe_for_G(fiber_path, fiber=0):
    files = glob.glob(f"{fiber_path}/fib/FIP_DataG*.csv")
    filename = max(files, key=num_rows)
    df = pd.read_csv(filename, header=None)
    zero_columns = (df == 0).all(axis=0)
    if zero_columns.any():
        logging.warning(
            f"CSV file '{filename}' contains {zero_columns.sum()} column(s) with all zeros. "
            f"Removing these columns to address a bug in older FIP_DataIso.csv files."
        )
        df = df.loc[:, ~zero_columns]
        df.columns = range(len(df.columns))
    df = df.rename(columns={0: "time_fip"})
    return df


def plot_pulse_differences(timestamps_raw, dt_raw, dt_fip, output_dir):
    """Plot pulse differences between RAW and FIP data.

    Parameters
    ----------
    timestamps_raw : np.ndarray
        Raw timestamps from HARP
    dt_raw : np.ndarray
        Differences between consecutive raw timestamps
    dt_fip : np.ndarray
        Differences between consecutive FIP timestamps
    output_dir : str
        Directory to save the plot

    Returns
    -------
    str
        Path to the saved plot
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
    ax[0].plot(np.cumsum(dt_raw), dt_raw, lw=0.5, marker=".", ms=2)
    ax[0].set_title("RAW")
    ax[1].plot(np.cumsum(dt_fip), dt_fip, lw=0.5, marker=".", ms=2)
    ax[1].set_ylabel("Inter-pulse interval [s]", y=1.2)
    ax[1].set_title("FIP")
    ax[1].set_xlabel("Time [s]")
    T = max(np.sum(dt_raw), np.sum(dt_fip))
    ax[1].set_xlim(-0.01 * T, 1.01 * T)
    plt.tight_layout(pad=0.4)

    plot_path = f"{output_dir}/pulse_diff.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def plot_g0_trace(df_G, nwbfile, dt_fip, output_dir):
    """Plot G0 traces before and after alignment.

    Parameters
    ----------
    df_G : pd.DataFrame
        Dataframe containing the G0 data from FIP
    nwbfile : NWBFile
        NWB file containing aligned data
    dt_fip : np.ndarray
        Differences between consecutive FIP timestamps
    output_dir : str
        Directory to save the plot

    Returns
    -------
    str
        Path to the saved plot
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
    ax[0].plot(np.cumsum(dt_fip), df_G[1][1:], lw=0.5, marker=".", ms=2)
    ax[0].set_title("from FIP, pre alignment")
    if nwbfile.acquisition.get("G_0") is not None:
        ax[1].plot(
            nwbfile.acquisition["G_0"].timestamps[:]
            - nwbfile.acquisition["G_0"].timestamps[0],
            nwbfile.acquisition["G_0"].data[:],
            lw=0.5,
            marker=".",
            ms=2,
        )
    ax[1].set_ylabel("CMOS pixel value of G0", y=1.2)
    ax[1].set_xlabel("Time [s]")
    ax[1].set_title("from NWB, post alignment")
    T = np.sum(dt_fip)
    ax[1].set_xlim(-0.01 * T, 1.01 * T)
    plt.tight_layout(pad=0.4)

    plot_path = f"{output_dir}/G0_trace.png"
    plt.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def plot_gocue_peth(timestamps_raw, df_G, nwbfile, GoCueTime, output_dir):
    """Plot peri-event time histograms around GoCue events.

    Parameters
    ----------
    timestamps_raw : np.ndarray
        Raw timestamps from HARP
    df_G : pd.DataFrame
        Dataframe containing G data from FIP
    nwbfile : NWBFile
        NWB file containing aligned data
    GoCueTime : list
        List of GoCue timestamps
    output_dir : str
        Directory to save the plot

    Returns
    -------
    str
        Path to the saved plot
    tuple
        (plot_path, p_values, significant)
    """
    fig, ax = plt.subplots(4, 2, figsize=(8, 8), sharex=True)
    p_values = np.full(4, np.nan)

    for fib in range(4):
        for i in (0, 1):
            try:
                timestamps = (
                    timestamps_raw
                    if i == 0
                    else nwbfile.acquisition[f"G_{fib}"].timestamps[:]
                )
                data = (
                    df_G[fib + 1] if i == 0 else nwbfile.acquisition[f"G_{fib}"].data[:]
                )
                inds = [
                    np.argmin(np.abs(timestamps - go))
                    for go in GoCueTime
                    if (go >= timestamps[10] and go <= timestamps[-50])
                ]  # being lazy but quick, interpolating would be better
                if len(inds) == 0:
                    raise Exception("No GoCue occurs within timestamps")
                snippets = np.array(
                    [data[i - 10 : i + 50] for i in inds if i <= len(data) - 50]
                )
                snippets -= snippets[:, :10].mean(1)[:, None]
                t = np.arange(-10, 50) * 0.05
                ax[fib, i].plot([-0.5, 0], [0, 0], alpha=0.7, color="gray")
                mu = snippets[:, 10:20].mean()
                sem = snippets[:, 10:20].mean(1).std(ddof=1) / np.sqrt(len(snippets))
                ax[fib, i].fill_between(
                    [0, 0.5], mu - sem, mu + sem, alpha=0.5, color="C4"
                )
                ax[fib, i].plot([0, 0.5], [mu, mu], alpha=0.7, color="C4")
                ax[fib, i].fill_between(
                    t,
                    snippets.mean(0) - snippets.std(0, ddof=1) / np.sqrt(len(snippets)),
                    snippets.mean(0) + snippets.std(0, ddof=1) / np.sqrt(len(snippets)),
                    alpha=0.5,
                    color=f"C{fib}",
                )
                ax[fib, i].plot(t, snippets.mean(0), lw=3, c=f"C{fib}")
                p_value = ttest_1samp(snippets[:, 10:20].mean(1), popmean=0).pvalue
                if i:
                    p_values[fib] = p_value
                ax[fib, i].text(
                    0.95,
                    0.95,
                    f"p = {p_value:.2e}" if p_value < 0.001 else f"p = {p_value:.4f}",
                    ha="right",
                    va="top",
                    transform=ax[fib, i].transAxes,
                    color="C4",
                )
            except Exception as e:
                logging.error(f"Error during GoCue plot: {e}")
            ax[fib, 0].set_ylabel(f"ROI {fib}")
            ax[3, i].set_xlabel("Time from Go Cue [s]")
            ax[0, i].set_title(
                ("from RAW and FIP, pre alignment", "from NWB, post alignment")[i]
            )
    plt.tight_layout(pad=0.4, w_pad=1.5)

    plot_path = f"{output_dir}/GoCue_PETH.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    significant = np.any(p_values < 0.05)
    if not significant:
        logging.error(
            "\033[91mOne-sample t-test yields no significant difference between "
            "the mean activity 500 ms before vs. 500 ms after GoCue.\033[0m"
        )

    return plot_path, p_values, significant


def create_plot_metric(
    name, reference, description="", status=Status.PENDING, value=None
):
    """Create a QC metric for a plot.

    Parameters
    ----------
    name : str
        The name of the metric.
    reference : str
        Path to the reference image for this metric.
    description : str, optional
        Description of the metric. Defaults to empty string.
    status : Status, optional
        The status to assign to this metric (PASS, FAIL, or PENDING).
        Defaults to Status.PENDING.

    Returns
    -------
    QCMetric
        The created quality control metric.
    """
    return QCMetric(
        name=name,
        reference=reference.replace("/results/", ""),
        description=description,
        status_history=[
            QCStatus(
                evaluator="Pending review" if status == Status.PENDING else "Automatic",
                timestamp=dt.now(),
                status=status,
            )
        ],
        value=(
            DropdownMetric(
                options=["Plot looks good", "Plot shows issues"],
                status=[Status.PASS, Status.FAIL],
            )
            if status == Status.PENDING
            else value
        ),
    )


def create_p_value_metric(p_values):
    """Create a QC metric for the p-value significance test.

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values from the t-test.

    Returns
    -------
    QCMetric
        The created quality control metric.
    """
    significant = np.any(p_values < 0.05)
    return QCMetric(
        name="Significant response to Go Cue",
        description=(
            "Check for a significant difference in mean activity "
            "between the 500 ms before and after the Go Cue."
        ),
        status_history=[
            QCStatus(
                evaluator="Automatic",
                timestamp=dt.now(),
                status=Status.PASS if significant else Status.FAIL,
            )
        ],
        value={f"ROI {i}": p for i, p in enumerate(p_values)},
    )


def create_truncation_metric(drop_start, drop_end):
    """Create a QC metric for the truncated timestamps.

    Parameters
    ----------
    drop_start : int
        Number of timestamps truncated at start.
    drop_end : int
        Number of timestamps truncated at end.

    Returns
    -------
    QCMetric
        The created quality control metric.
    """
    return QCMetric(
        name="Truncated timestamps",
        description="Number of HARP timestamps truncated at start & end for alignment",
        status_history=[
            QCStatus(
                evaluator="Automatic",
                timestamp=dt.now(),
                status=Status.PASS,
            )
        ],
        value={"start": drop_start, "end": drop_end},
    )


def create_gap_metric(gaps):
    """Create a QC metric for the gaps retained in timestamps.

    Parameters
    ----------
    gaps : list
        List of retained gaps.

    Returns
    -------
    QCMetric
        The created quality control metric.
    """
    return QCMetric(
        name="Retained gaps in timestamps",
        description="Retained gaps in timestamps because they occurred mid-session",
        status_history=[
            QCStatus(
                evaluator="Automatic",
                timestamp=dt.now(),
                status=Status.PASS if gaps == [] else Status.FAIL,
            )
        ],
        value=(
            None
            if gaps is None
            else [
                f"Gap between index {i} and {i+1} of size {s:.4f}s" for (i, s) in gaps
            ]
        ),
    )


def create_time_metric(fiber_path, df_G):
    """Create a QC metric for FIP start & end time.

    Parameters
    ----------
    fiber_path : str
        The path to the directory where the FIP files are located
    df_G : pd.DataFrame
        DataFrame containing G data from FIP

    Returns
    -------
    QCMetric
        The created quality control metric.
    """
    for file in [
        f for f in glob.glob(f"{fiber_path}/behavior/*.json") if "model" not in f
    ]:
        with open(file, "r") as jsonFile:
            values = json.load(jsonFile)
    start = float(
        df_G["time_fip"].values[0] / 1000
        - to_daytime(values.get("fiber_photometry_start_time", np.nan))
    )
    end = float(
        df_G["time_fip"].values[-1] / 1000
        - to_daytime(values.get("fiber_photometry_end_time", np.nan))
    )
    if start >= 0.9 and start <= 1.2 and end >= 0 and end <= 1.2:
        status = Status.PASS
    else:
        status = Status.FAIL
        s = start < 0.9 or start > 1.2
        logging.error(
            f"\033[91mUnusual time difference: {'first' if s else 'last'} CSV timestamp "
            f"vs. JSON `fiber_photometry_{'start' if s else 'end'}_time`.\033[0m"
        )

    return QCMetric(
        name="Time difference between FIP start & end times in CSV vs JSON",
        description="Typically (first CSV timestamp - JSON start_time) is in [0.9, 1.2] "
        "and (last CSV timestamp - JSON end_time) is in [0, 1.2]",
        status_history=[
            QCStatus(
                evaluator="Automatic",
                timestamp=dt.now(),
                status=status,
            )
        ],
        value={"start": f"{start:.3f}s", "end": f"{end:.3f}s"},
    )


def run_qc(drop_start=None, drop_end=None, gaps=None):
    # Setup
    fiber_path = "/data/fiber_raw_data"
    nwb_path = glob.glob("/results/nwb/*")[0]
    with open("/data/fiber_raw_data/data_description.json") as f:
        name = json.load(f).get("name")
    if name is None:
        name = Path(
            [f for f in glob.glob(f"{fiber_path}/behavior/*.json") if "model" not in f][
                0
            ]
        ).stem
    output_dir = "/results/alignment-qc"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    GoCueTime = get_GoCueTime(fiber_path)
    timestamps_raw = timestamps_from_harpraw(fiber_path)
    df_G = dataframe_for_G(fiber_path)
    dt_raw = np.diff(timestamps_raw)
    dt_fip = np.diff(df_G["time_fip"]) / 1000

    with NWBZarrIO(path=nwb_path, mode="r") as io:
        nwbfile = io.read()

    # Generate plots
    pulse_diff_path = plot_pulse_differences(timestamps_raw, dt_raw, dt_fip, output_dir)
    g0_trace_path = plot_g0_trace(df_G, nwbfile, dt_fip, output_dir)
    gocue_path, p_values, significant = plot_gocue_peth(
        timestamps_raw, df_G, nwbfile, GoCueTime, output_dir
    )

    # Create metrics for QC evaluation
    if gaps is None:
        stat, val = Status.PENDING, None
    elif gaps == []:
        stat, val = Status.PASS, "No gaps retained"
    else:
        ng = len(gaps)
        stat, val = Status.FAIL, f"{ng} gap{'' if ng==1 else 's'} retained"
    metrics = [
        create_plot_metric(
            "Inter-pulse Intervals",
            f"{pulse_diff_path}",
            "Comparison of inter-pulse intervals between RAW and FIP data",
            Status.PASS,
        ),
        create_plot_metric(
            "Trace Comparison (green channel, ROI 0)",
            f"{g0_trace_path}",
            "Comparison of G0 traces before and after alignment",
            stat,
            val,
        ),
        create_plot_metric(
            "GoCue PETH (Peri-event time histogram)",
            f"{gocue_path}",
            "Peri-event time histograms around GoCue events",
            Status.PASS if significant else Status.FAIL,
            (
                f"{'S' if significant else 'No s'}ignificant "
                "activity difference pre/post Go Cue"
            ),
        ),
        create_p_value_metric(p_values),
        create_truncation_metric(drop_start, drop_end),
        create_gap_metric(gaps),
        create_time_metric(fiber_path, df_G),
    ]

    # Create evaluation and save QC report
    evaluation = QCEvaluation(
        name="Alignment check",
        modality=Modality.FIB,
        stage=Stage.PROCESSING,
        metrics=metrics,
        description=(
            "Quality control of fiber photometry data alignment. "
            "Verifies proper alignment of timestamps and "
            "presence of significant response to Go Cue."
        ),
        latest_status=Status.PASS if significant else Status.FAIL,
    )

    qc = QualityControl(evaluations=[evaluation])
    qc.write_standard_file(output_directory=output_dir)


if __name__ == "__main__":
    run_qc()
