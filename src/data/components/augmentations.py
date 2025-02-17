import cv2
import numpy as np
import albumentations as A


class ResizeWithAspectRatio(A.core.transforms_interface.DualTransform):
    """
    Custom Albumentations transform to resize an image to a target width and height
    while maintaining the aspect ratio and adding padding.
    """

    def __init__(self, target_width, target_height, always_apply=True, p=1.0):
        """
        Args:
            target_width (int): Desired width of the output image.
            target_height (int): Desired height of the output image.
            fill_color (tuple): Color to use for padding (default: black).
            always_apply (bool): Always apply the transformation.
            p (float): Probability of applying the transformation.
        """
        super().__init__(always_apply, p)
        self.target_width = target_width
        self.target_height = target_height

    def resize_keeping_aspect_ratio(self, image, padding_color=(0, 0, 0)):
        """
        Resize an image to the target width and height while keeping the aspect ratio
        by adding padding. Uses OpenCV for processing.

        Args:
            image (numpy.ndarray): Input image as a NumPy array (H x W x C).
            target_width (int): Desired width of the output image.
            target_height (int): Desired height of the output image.
            padding_color (tuple): Color for padding in (B, G, R) format. Default is black.

        Returns:
            numpy.ndarray: Resized image with padding, maintaining aspect ratio.
        """
        # Get the original image dimensions
        original_height, original_width = image.shape[:2]

        # Calculate the aspect ratios
        aspect_ratio_original = original_width / original_height
        aspect_ratio_target = self.target_width / self.target_height

        # Determine the new dimensions while maintaining aspect ratio
        if aspect_ratio_original > aspect_ratio_target:
            # Width is the limiting factor
            new_width = self.target_width
            new_height = int(self.target_width / aspect_ratio_original)
        else:
            # Height is the limiting factor
            new_height = self.target_height
            new_width = int(self.target_height * aspect_ratio_original)

        # Resize the image using NEAREST interpolation
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

        # Calculate padding to center the resized image
        delta_width = self.target_width - new_width
        delta_height = self.target_height - new_height
        top = delta_height // 2
        bottom = delta_height - top
        left = delta_width // 2
        right = delta_width - left

        # Add padding to the resized image
        padded_image = cv2.copyMakeBorder(
            resized_image, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=padding_color
        )

        return padded_image

    def apply(self, img, **params):
        return self.resize_keeping_aspect_ratio(image=img, padding_color=(0, 0, 0))

    def apply_to_mask(self, img, **params):
        return self.resize_keeping_aspect_ratio(image=img, padding_color=0)