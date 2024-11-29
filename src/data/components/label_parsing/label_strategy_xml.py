import numpy as np
import xml.etree.ElementTree as ET

from .label_strategy import LabelStrategy


class XmlBboxLabelStrategy(LabelStrategy):
    def process_label(self, label_path: str, image_shape: tuple) -> np.array:
        """Converts xml bbox information into grayscale image.

        Args:
            label_path (str): _description_
            image_shape (tuple): Corresponding image shape.

        Returns:
            np.array: The processed label data as image.
        """
        # Parse the XML file
        tree = ET.parse(label_path)
        root = tree.getroot()
        # Create a blank label image based on the input image shape
        label_image = np.zeros(image_shape[:2], dtype=np.uint8)

        # Extract bounding boxes and draw them on the label image
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Fill the bounding box area with white (255)
            label_image[ymin:ymax, xmin:xmax] = 255

        return label_image
