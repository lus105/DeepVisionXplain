import os
import shutil
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
XML_EXTENSION = ".xml"
JSON_EXTENSION = ".json"


def get_file_paths_rec(root_path: str, file_extensions: list = None) -> list:
    """Returns a list of file paths from the given directory and its subdirectories.

    Args:
        root_path (str): Path to the root directory.
        file_extensions (list, optional): List of file extensions to filter by. Defaults to None.

    Returns:
        list: List of file paths matching the specified extensions.
    """
    if not os.path.exists(root_path):
        log.warning(f"Directory does not exist: {root_path}")
        return []

    if file_extensions is None:
        log.warning("No extensions provided!")
        return []

    file_paths = [
        os.path.join(root, file)
        for root, dirs, files in os.walk(root_path)
        for file in files
        if os.path.splitext(file)[1].lower() in file_extensions
    ]

    return file_paths

def save_files(file_paths: list, target_dir: str):
    """Copies files from the provided list of file paths to a target directory.

    Args:
        file_paths (list): A list of file paths to be copied.
        target_dir (str): The destination directory where files will be copied to.
    """
    os.makedirs(target_dir, exist_ok=True)

    # Check if the directory is empty. If not, raise an exception.
    if os.listdir(target_dir):
        log.warning(f"Directory {target_dir} is not empty. Skipping saving.")
        return

    for f in file_paths:
        shutil.copy(f, target_dir)

def get_file_name(path: str) -> str:
    """
    Extracts the file name from a given file path, excluding the extension.

    Args:
    - path (str): The full file path.

    Returns:
    - str: The name of the file without its extension.
    """
    file_name_with_ext = os.path.basename(path)
    file_name, _ = os.path.splitext(file_name_with_ext)
    return file_name


def get_file_extension(path: str) -> str:
    """
    Retrieves the file extension from a given file path.

    Args:
    - path (str): The full file path.

    Returns:
    - str: The extension of the file.
    """
    _, file_extension = os.path.splitext(path)
    return file_extension


def find_annotation_file(directory: str, file_name: str):
    """
    Searches for a file with a specified base name and various image extensions in a given directory.

    Args:
    - directory (str): The directory to search in.
    - file_name (str): The base name of the file to find.

    Returns:
    - str or None: The path of the found file with the specified base name, or None if not found.
    """
    file_extensions = IMAGE_EXTENSIONS + XML_EXTENSION + JSON_EXTENSION
    for extension in file_extensions:
        file_path = os.path.join(directory, file_name + extension)
        if os.path.isfile(file_path):
            return file_path
    return None


def clear_directory(directory_path):
    """
    Clear all files and subdirectories within a directory.

    Args:
    - directory_path (str): The path to the directory to be cleared.
    """
    if not os.path.isdir(directory_path):
        return
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            clear_directory(file_path)
            os.rmdir(file_path)
