from src.data.components.utils import IMAGE_EXTENSIONS, XML_EXTENSION
from .label_strategy import LabelStrategy
from .label_strategy_xml import XmlBboxLabelStrategy
from .label_strategy_image import ImageLabelStrategy


def get_label_strategy(file_extension: str) -> LabelStrategy:
    """Returns an instance of LabelStrategy based on the file extension.

    Args:
        file_extension (str): Extension of a label file.

    Raises:
        ValueError: If no label strategy is available for extension.

    Returns:
        LabelStrategy: Class  for processing labels.
    """
    for extension in IMAGE_EXTENSIONS:
        if extension == file_extension:
            return ImageLabelStrategy()

    if file_extension == XML_EXTENSION:
        return XmlBboxLabelStrategy()
    else:
        raise ValueError(f'No label strategy available for extension {file_extension}')
