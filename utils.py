import os
import subprocess


def maybe_download_tarball_with_pget(
    url: str,
    dest: str,
):
    """
    Downloads a tarball from url and decompresses to dest if dest does not exist. Remote path is constructed
    by concatenating remote_path and remote_filename. If remote_path is None, files are not downloaded.

    Args:
        url (str): URL to the tarball
        dest (str): Path to the directory where the tarball should be decompressed

    Returns:
        path (str): Path to the directory where files were downloaded

    """
    
    if not os.path.exists(dest):
        print("Downloading tarball...")
        command = ["pget", url, dest, "-x"]
        subprocess.check_call(command, close_fds=True)
    
    return dest
