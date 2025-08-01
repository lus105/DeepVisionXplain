from collections.abc import Sequence
import PIL.Image
import torch
from torch import nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F
from torchvision.transforms.functional import get_image_size


class ResizeAndExtrapolateBorders(nn.Module):
    """
    Resize the input image only if it's larger than the destination size (preserving aspect ratio),
    and then pad it with either replicated borders or a constant color to fit exactly into (H, W).
    """

    def __init__(
        self,
        size: tuple[int, int],
        replicate_borders: bool = True,
        inpaint_color: tuple[int, int, int] = (0, 0, 0),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: bool = True,
    ):
        """
        Args:
            size (tuple): (height, width) of the destination image.
            replicate_borders (bool): Whether to use border replication or constant fill.
            inpaint_color (tuple): RGB color used when replicate_borders=False.
            interpolation (InterpolationMode): Interpolation mode for resizing.
            antialias (bool): Whether to use antialiasing during resizing (only for tensors).
        """
        super().__init__()
        if isinstance(size, Sequence) and len(size) != 2:
            raise ValueError('If size is a sequence, it should have 2 values')
        self.dest_h, self.dest_w = size
        self.replicate_borders = replicate_borders
        self.inpaint_color = tuple(inpaint_color)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(
        self, img: torch.Tensor | PIL.Image.Image
    ) -> torch.Tensor | PIL.Image.Image:
        """
        Resizes and pads an input image or tensor to fit the target dimensions
        while preserving aspect ratio.
        If the input image is larger than the target size (`self.dest_w`, `self.dest_h`),
        it is resized to fit within the target dimensions, maintaining aspect ratio.
        The resized image is then padded to exactly match the target size.
        Padding can be applied using either border replication or a constant fill color.
        Args:
            img (torch.Tensor | PIL.Image.Image): The input image or tensor to be transformed.
        Returns:
            torch.Tensor | PIL.Image.Image: The resized and padded image or tensor.
        """
        orig_w, orig_h = get_image_size(img)

        # Resize only if larger
        if orig_w > self.dest_w or orig_h > self.dest_h:
            x_ratio = self.dest_w / orig_w
            y_ratio = self.dest_h / orig_h
            resize_ratio = min(x_ratio, y_ratio)
            new_w = int(orig_w * resize_ratio)
            new_h = int(orig_h * resize_ratio)
            img = F.resize(
                img,
                [new_h, new_w],
                interpolation=self.interpolation,
                antialias=self.antialias,
            )
        else:
            new_w, new_h = orig_w, orig_h

        # Compute padding
        pad_w = self.dest_w - new_w
        pad_h = self.dest_h - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - (pad_w // 2), pad_h - (pad_h // 2))

        if self.replicate_borders:
            padding_mode = 'edge'
            fill = None
        else:
            padding_mode = 'constant'
            fill = self.inpaint_color

        return F.pad(img, padding, fill=fill, padding_mode=padding_mode)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(size=({self.dest_h}, {self.dest_w}), '
            f'replicate_borders={self.replicate_borders}, inpaint_color={self.inpaint_color}, '
            f'interpolation={self.interpolation.value}, antialias={self.antialias})'
        )
