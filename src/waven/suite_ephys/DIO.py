"""Trodes digital-input discovery and readers.

Recording exports do not consistently name their digital-input directory
``*.DIO``.  The durable signal of a usable directory is instead a Trodes
``*Din<port>.dat`` file, so discovery is based on files rather than a folder
name.
"""
import re
from pathlib import Path
import numpy as np
from .readTrodesExtractedDataFile3 import readTrodesExtractedDataFile


_DIN_FILE_RE = re.compile(r"din(?P<port>\d+)\.dat$", re.IGNORECASE)
_PART_RE = re.compile(r"(?:^|[._\\/-])part(?P<part>\d+)(?=$|[._\\/-])", re.IGNORECASE)


def _dio_directory_records(rec_folder):
    """Return directories below ``rec_folder`` and the digital ports in each."""
    rec_folder = Path(rec_folder)
    if not rec_folder.is_dir():
        raise FileNotFoundError(f"Raw ephys data folder does not exist: {rec_folder}")

    records = {}
    # Include the selected folder itself as some exports place Din*.dat files
    # directly in the recording directory.  ``rglob`` then covers arbitrarily
    # named nested recording/part folders without relying on a ``.DIO`` suffix.
    try:
        files = [
            path for path in rec_folder.rglob("*")
            if path.is_file() and path.suffix.lower() == ".dat"
        ]
    except OSError as exc:
        raise OSError(f"Could not scan raw ephys data folder for .dat files: {rec_folder}") from exc

    for path in files:
        match = _DIN_FILE_RE.search(path.name)
        record = records.setdefault(path.parent, {"ports": set(), "dat_files": []})
        record["dat_files"].append(path)
        if match:
            record["ports"].add(int(match.group("port")))
    return records


def _dio_sort_key(folder, root):
    """Order split recording parts before the stable relative folder name."""
    try:
        relative = folder.relative_to(root).as_posix().lower()
    except ValueError:
        relative = folder.as_posix().lower()
    match = _PART_RE.search(relative)
    part = int(match.group("part")) if match else 1
    return part, relative


def available_dio_ports(rec_folder):
    """Return all Trodes digital-input ports discovered below a raw-data folder."""
    records = _dio_directory_records(rec_folder)
    return tuple(sorted({port for record in records.values() for port in record["ports"]}))


def get_dio_folders(rec_folder, channel_id=None):
    """Find folders containing Trodes ``Din<port>.dat`` files.

    Folder names are intentionally ignored.  A selected ``channel_id`` filters
    the result to the recording parts that actually contain that port, allowing
    split recordings to be concatenated safely.
    """
    rec_folder = Path(rec_folder)
    records = _dio_directory_records(rec_folder)
    channel_id = None if channel_id is None else int(channel_id)
    matching = [
        folder
        for folder, record in records.items()
        if record["ports"] and (channel_id is None or channel_id in record["ports"])
    ]
    if matching:
        return sorted(matching, key=lambda folder: _dio_sort_key(folder, rec_folder))

    available_ports = tuple(
        sorted({port for record in records.values() for port in record["ports"]})
    )
    generic_dat_folders = sorted(
        str(folder) for folder, record in records.items() if record["dat_files"]
    )
    if channel_id is None:
        detail = (
            f" .dat files were found in: {', '.join(generic_dat_folders[:5])}"
            if generic_dat_folders
            else ""
        )
        raise FileNotFoundError(
            "No Trodes digital-input files ending in Din<port>.dat were found beneath "
            f"{rec_folder}.{detail}"
        )
    ports_text = ", ".join(str(port) for port in available_ports) or "none"
    raise FileNotFoundError(
        f"No Trodes digital-input file ending in Din{channel_id}.dat was found beneath "
        f"{rec_folder}. Available ports: {ports_text}."
    )


def extract_DIN(DIO_folder, channel_id):
    """Read one port's timestamps and states from a discovered folder."""
    DIO_folder = Path(DIO_folder)
    channel_id = int(channel_id)
    suffix = f"din{channel_id}.dat"
    din_files = sorted(
        (file for file in DIO_folder.iterdir() if file.is_file() and file.name.lower().endswith(suffix)),
        key=lambda file: file.name.lower(),
    )
    if not din_files:
        raise FileNotFoundError(
            f"No Trodes digital-input file ending in Din{channel_id}.dat in {DIO_folder}"
        )

    din_file = din_files[0]
    # read the file
    time = readTrodesExtractedDataFile(din_file)['data']['time']
    state = readTrodesExtractedDataFile(din_file)['data']['state']
    return time, state


def concatenate_din_data(dio_folders, channel_id: int):
    """Concatenate a port's data across ordered recording parts."""
    if not dio_folders:
        raise ValueError("At least one digital-input folder is required.")
    time, state = extract_DIN(dio_folders[0], channel_id)

    if len(dio_folders) == 1:
        return time, state
    
    
    for i in range(1, len(dio_folders)):
        time_, state_ = extract_DIN(dio_folders[i], channel_id)
        # if the end of the last state is the same as the start of the current state, remove the first element of the current state and time
        if len(state) and len(state_) and state[-1] == state_[0]:
            state_ = state_[1:]
            time_ = time_[1:]

        time = np.concatenate((time, time_))
        state = np.concatenate((state, state_))
    return time, state
