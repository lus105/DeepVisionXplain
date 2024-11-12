import shutil
from pathlib import Path
from typing import Optional
from enum import Enum

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
XML_EXTENSION = ".xml"
JSON_EXTENSION = ".json"

class DatasetType(Enum):
    ImageLabelBinary = 1
    ImageLabelMultiClass = 2
    ImageBinary = 3
    ImageMultiClass = 4


def list_files(root_path: Path, file_extensions: list = None) -> list[Path]:
    """Returns a list of file paths from the given directory and its subdirectories.

    Args:
        root_path (Path): Path to the root directory.
        file_extensions (list, optional): List of file extensions to filter by. Defaults to None.

    Returns:
        list[Path]: List of file paths matching the specified extensions.
    """
    root = Path(root_path)
    
    if not root.exists():
        log.warning(f"Directory does not exist: {root_path}")
        return []

    if file_extensions is None:
        log.warning("No extensions provided!")
        return []

    # Convert extensions to a set of lowercase extensions for matching
    file_extensions = {ext.lower() for ext in file_extensions}
    
    file_paths = [
        file_path for file_path in root.rglob('*')
        if file_path.is_file() and file_path.suffix.lower() in file_extensions
    ]

    return file_paths

def list_dirs(root_path: Path) -> list[Path]:
    """Returns all directories in a given directory.

    Args:
        root_path (Path): Path to search for directories.

    Returns:
        list[Path]: List of found directories.
    """
    return [f for f in Path(root_path).iterdir() if f.is_dir()]

def save_files(file_paths: list[Path], target_dir: Path) -> None:
    """Copies files from the provided list of file paths to a target directory.

    Args:
        file_paths (list[Path]): A list of file paths to be copied.
        target_dir (Path): The destination directory where files will be copied to.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Check if the directory is empty
    if any(target_path.iterdir()):
        log.warning(f"Directory {target_dir} is not empty. Skipping saving.")
        return

    for file_path in file_paths:
        file_path = Path(file_path)  # Ensure each file path is a Path object
        shutil.copy(file_path, target_path / file_path.name)

def find_annotation_file(directory: Path, file_name: str, file_extensions: list = IMAGE_EXTENSIONS) -> Optional[Path]:
    """Searches for a file with a specified base name and various extensions in a given directory.

    Args:
        directory (Path): The directory to search in.
        file_name (str): The base name of the file to find.
        file_extensions (list, optional): List of file extensions to filter by. Defaults to IMAGE_EXTENSIONS.

    Returns:
        Optional[Path]: The path of the found file with the specified base name, or None if not found.
    """
    directory_path = Path(directory)

    for extension in file_extensions:
        file_path = directory_path / f"{file_name}{extension}"
        if file_path.is_file():
            return Path(file_path)

    return None

def clear_directory(directory_path: Path) -> None:
    """Clear all files and subdirectories within a directory.

    Args:
        directory_path (Path): The path to the directory to be cleared.
    """
    directory = Path(directory_path)
    
    if not directory.is_dir():
        return
    
    for item in directory.iterdir():
        if item.is_file():
            item.unlink()  # Removes the file
        elif item.is_dir():
            clear_directory(item)  # Recursively clear subdirectory
            item.rmdir()  # Removes the empty subdirectory
