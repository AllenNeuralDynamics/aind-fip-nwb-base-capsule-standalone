"""top level run script"""

import argparse
import glob
import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from aind_log_utils import log
from hdmf_zarr import NWBZarrIO
from pydantic import Field
from pydantic_settings import BaseSettings

from aind_nwb_utils.utils import create_base_nwb_file
from util.add_fiber_to_nwb import (add_fiber_data_to_nwb, deal_with_nans,
                                   get_fiber_data_by_channel)
from util.add_info_to_nwb import add_info_to_nwb
from util.aligned_fiber_to_nwb_legacy import (
    append_aligned_fiber_to_nwb,
    detect_fiber_photometry_system,
    num_rows,
)
from util.add_fiber_to_nwb import (
    get_fiber_data_by_channel,
    add_fiber_data_to_nwb,
    deal_with_nans,
)
from util.qc import run_qc


class FiberSettings(BaseSettings, cli_parse_args=True):
    """
    Settings for Fiber Photometry
    """

    input_directory: Path = Field(
        default=Path("/data"), description="Directory where data is"
    )
    output_directory: Path = Field(
        default=Path("/results/"), description="Output directory"
    )
    min_fip_duration: int = Field(
        default=6000,
        description="Minimum amount of FIP data required before FIP data is processed",
    )
    max_drop: int = Field(
        default=3600,
        description="Maximum number of samples to drop from the end if NaNs or a large gap in the time stamps are found near the end.",
    )
if __name__ == "__main__":
    settings = FiberSettings()

    logging.info(f"Fiber Settings, {settings.model_dump()}")
    fiber_fp = settings.input_directory / "fiber_raw_data"
    # Load subject data
    subject_json_path =  fiber_fp / "subject.json"
    with subject_json_path.open("r") as f:
        subject_data = json.load(f)

    subject_id = subject_data.get("subject_id", None)

    # Load data description
    data_description_path = fiber_fp / "data_description.json"
    with data_description_path.open("r") as f:
        date_data = json.load(f)

    session_path = fiber_fp / "session.json"
    with session_path.open("r") as f:
        session_data = json.load(f)
    date = session_data["session_start_time"]

    asset_name = date_data["name"]

    log.setup_logging(
        "aind-fib-nwb-base-capsule",
        mouse_id=subject_data["subject_id"],
        session_name=asset_name,
    )


    nwb_filename = f"{asset_name}.nwb"
    nwb_output_path = settings.output_directory / "nwb"
    nwb_output_path.mkdir(parents=True, exist_ok=True)

    base_nwb_file = create_base_nwb_file(fiber_fp)

    if not [i for i in fiber_fp.glob("fib/")]:
        raise ValueError("No fiber data detected")

    fiber_directories = [i for i in fiber_fp.glob("fib/*") if i.is_dir()]
    if fiber_directories:
        logging.info("Standard file format detected")
        fiber_channel_data = get_fiber_data_by_channel(fiber_directories[0])
        fiber_channel_data_cleaned_for_nans = deal_with_nans(fiber_channel_data)
        nwbfile, src_io = add_fiber_data_to_nwb(
            base_nwb_file, fiber_channel_data_cleaned_for_nans
        )
    # Append FIP to behavior NWB if FIP or fib exists
    else:  # legacy acquisition
        logging.info("Legacy file format detected")
        fiber_photometry_system, filenames = detect_fiber_photometry_system(fiber_fp)
        fip_duration = min([num_rows(f) for f in filenames])
        if fip_duration > settings.min_fip_duration:
            nwbfile, drop_start, drop_end, kept_gaps, behavior_system = append_aligned_fiber_to_nwb(
                fiber_fp, settings.max_drop, base_nwb_file
            )
            logging.info("Successfully appended the aligned fiber photometry data.")
        else:
            raise ValueError(
                f"FIP data is present, but only {fip_duration / 20}s long. "
                f"This is shorter than the required {settings.min_fip_duration / 20}s, "
                "thus treating dataset as behavior-only instead of appending FIP data."
            )

    nwbfile = add_info_to_nwb(nwbfile, fiber_fp)
    nwb_output_fn = nwb_output_path / nwb_filename

    with NWBZarrIO(path=nwb_output_fn.as_posix(), mode="w") as io:
        io.write(nwbfile)

        logging.info("Successfully wrote NWB file.")
    
    if "drop_start" in locals():
        if behavior_system != "pavlovian":
            run_qc(drop_start, drop_end, kept_gaps)
        elif behavior_system == "pavlovian":
            logging.info("No legacy fiber data to qc")
            os.mkdir(os.path.join("/results", "alignment-qc"))
            qc_file_path = Path("/results") / "alignment-qc" / "no_fip_to_qc.txt"
            # Create an empty file
            with open(qc_file_path, "w") as file:
                file.write(
                    "This is a pavlovian session"
                )
    else: 
        logging.info("No legacy fiber data to qc")
        os.mkdir(os.path.join("/results", "alignment-qc"))
        qc_file_path = Path("/results") / "alignment-qc" / "no_fip_to_qc.txt"
        # Create an empty file
        with open(qc_file_path, "w") as file:
            file.write(
                "This is not a fiber legacy session"
            )
