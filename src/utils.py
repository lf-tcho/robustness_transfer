from pathlib import Path
import shutil
import requests
import zipfile
from tqdm import tqdm


def download_url(url, save_path, chunk_size=128):
    """Download file from url.

    :param url: URL to download from
    :param save_path: Path to save download to
    :param chunk_size: Number of bytes to read into memory
    """
    request = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in request.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def unzip(source, destination):
    """Extract all for a zip file.

    :param source: Path to zip file
    :param destination: Output folder
    """
    with zipfile.ZipFile(source, "r") as zip_ref:
        for member in tqdm(
            zip_ref.infolist(), desc=f"Extracting {source} to {destination}"
        ):
            zip_ref.extract(member, destination)


def copy_all_files(source, destination):
    """Copy all files from source to destination

    :param source: Source folder
    :param destination: Destination folder
    """
    for src_file in Path(source).glob("*.*"):
        shutil.copy(src_file, destination)


def get_experiment_name(arg_dict):
    """Create experiment name.

    :param arg_dict: Dictionary of argument names and respective values
    """
    return "_".join([f"{key}_{arg}" for key, arg in sorted(list(arg_dict.items()))])
    