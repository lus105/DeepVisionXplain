from src.data.components.helper_utils import IMAGE_EXTENSIONS
from .label_strategy import LabelStrategy
from .xml_label_strategy import XmlLabelStrategy
from .image_label_strategy import ImageLabelStrategy

def get_label_strategy(file_extension: str) -> LabelStrategy:
    """
    Returns an instance of LabelStrategy based on the file extension.
    """
    for extension in IMAGE_EXTENSIONS:
        if extension == file_extension:
            return ImageLabelStrategy()
        
    if file_extension == '.xml':
        return XmlLabelStrategy()
    else:
        raise ValueError(f"No label strategy available for extension {file_extension}")