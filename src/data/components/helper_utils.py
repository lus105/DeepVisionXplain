import glob
import os
import shutil

def gather_image_from_dir(input_dir: str):
    """
    Collects a list of image file paths from a specified directory.

    Args:
    - input_dir (str): The directory from which to gather image files.

    Returns:
    - list: A list of file paths for images in the specified directory.
    Supported image formats include BMP, JPG, and PNG.
    """
    image_extensions = ['*.bmp', '*.jpg', '*.png']
    image_list = []
    for image_extension in image_extensions:
        image_list.extend(glob.glob(input_dir + image_extension))
    image_list.sort()
    return image_list

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
    image_extensions = ['.bmp', '.jpg', '.png', '.json']
    for image_extension in image_extensions:
        file_path = os.path.join(directory, file_name + image_extension)
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

def get_image_paths(directory_path: str):
    """
    Returns a list of image paths from the given directory and its subdirectories.

    Args:
    - directory_path (str): Path to the root directory.

    Returns:
    - list: List of image paths. Supported image formats include JPG, JPEG, PNG, and BMP.
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = [os.path.join(root, file) 
                   for root, dirs, files in os.walk(directory_path) 
                   for file in files 
                   if os.path.splitext(file)[1].lower() in image_extensions]
    
    return image_paths

def save_files(file_paths, target_dir):
    """
    Copies files from the provided list of file paths to a target directory.

    Args:
    - file_paths (list): A list of file paths to be copied.
    - target_dir (str): The destination directory where files will be copied to.
    """
    os.makedirs(target_dir, exist_ok=True)
    for f in file_paths:
        shutil.copy(f, target_dir)