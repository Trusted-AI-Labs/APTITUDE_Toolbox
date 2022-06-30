"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

import cv2
import numpy as np
import urllib.request
import ast
import logging
from typing import Tuple

log = logging.getLogger("aptitude-toolbox")

try:
    from turbojpeg import TurboJPEG

    tjpeg = TurboJPEG()
except:
    tjpeg = None
    log.warning("TurboJPEG could not be found, using cv2 to decode images instead.")


def get_cv2_img_from_str(path: str, flags=cv2.IMREAD_COLOR) -> np.array:
    """
    Decodes an image from a path and returns a np.array containing the image, under cv2 format.
    If installed and if it is a .jpg or .jpeg image, uses TurboJPEG instead for faster reading.

    Args:
        path (str): Path to the image to be decoded
        flags (int): A cv2 flag indicating the reading mode. By default, it uses cv2.IMREAD_COLOR.

    Returns:
        np.array: The image that was read from the input path.
    """
    with open(path, 'rb') as buffer:
        if tjpeg is not None and (path.endswith(".jpg") or path.endswith(".jpeg")):
            return tjpeg.decode(buffer.read())
        else:
            nparr = np.frombuffer(buffer.read(), dtype=np.uint8)
            return cv2.imdecode(nparr, flags)


def get_cv2_img_from_url(url, flags=cv2.IMREAD_COLOR) -> np.array:
    """
    Decodes an image from an URL and returns a np.array containing the image, under cv2 format.

    Args:
        url (str): A string containing the URL where the image should be fetched.
        flags (int): A cv2 flag indicating the reading mode. By default, it uses cv2.IMREAD_COLOR.

    Returns:
        np.array: The image that was read from the URL.
    """
    req = urllib.request.Request(url)
    return _get_cv2_img_from_buffer(urllib.request.urlopen(req), flags)


def _get_cv2_img_from_buffer(buffer, flags=cv2.IMREAD_COLOR):
    nparr = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(nparr, flags)


def resize(image: np.array, width: int, height: int) -> np.array:
    """
    Applies a resizing method on the image with cv2.INTER_AREA for the interpolation.

    Args:
        image: The image to be resized.
        width: The width of the image after resizing, in pixels.
        height: The height of the image after resizing, in pixels.

    Returns:
        np.array: A new image resized to the required dimension.

    """
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def get_roi_file(roi_path: str):
    """
    Gets the Region of Interest (ROI) from a path to an image file.

    Args:
        roi_path (str): The path to the image.

    Returns:
        np.array: A binary mask where white pixels represent the Region of Interest (ROI)\
        and the black pixels represent the regions to be ignored.
    """
    return get_cv2_img_from_str(roi_path, flags=cv2.IMREAD_COLOR)


def get_roi_coords(image_shape: tuple, roi_coords: str) -> np.array:
    """
    Gets the Region of Interest (ROI) from a set of polygon coords.

    Args:
        image_shape (tuple): The dimension of the binary mask (the image) that will be returned
        roi_coords (np.array): The set of the polygon coords that defines the Region of Interest (the white pixels).
            It must be of the following format: "(0, 0), (450, 0), (450, 200), (0, 200)"

    Returns:
        np.array: A binary mask where white pixels represent the Region of Interest (ROI)\
        and the black pixels represent the regions to be ignored.
    """
    roi_coords = ast.literal_eval(roi_coords)
    for c in roi_coords:
        assert c[0] <= image_shape[1] and c[1] <= image_shape[0], \
            "The provided coords (W, H: {}) are outside the image shape (W, H: {})" \
            .format((c[0], c[1]), (image_shape[1], image_shape[0]))

    roi = np.zeros(image_shape, dtype=np.uint8)
    polygon = np.array([roi_coords], dtype=np.int32)

    # get number of channel or if absent, default is 1
    num_frame_channels = image_shape[2] if len(image_shape) == 3 else 1
    mask_ignore_color = (255,) * num_frame_channels
    return cv2.fillPoly(roi, polygon, mask_ignore_color)


def apply_roi(image: np.array, roi: np.array) -> np.array:
    """
    Applies the Region of Interest (ROI), which is a binary mask, onto the image.

    Args:
        image (np.array): The original image on which the mask will be applied.
        roi (np.array): The mask to be applied on the image

    Returns:
        np.array: A new image where black pixels of the mask are applied on the image.
    """
    assert image.shape[:2] == roi.shape[:2], \
        "The mask image has not the same width or height as the frame to be masked."
    return cv2.bitwise_and(image, roi)


def add_borders(image: np.array, centered=False) -> Tuple[np.array, np.array]:
    """Adds black border to 'image' to keep the aspect ratio.
    returns the frame in letterbox format and the number of black pixels on each side.

    Args:
        image (np.array): The image to apply the transformation.
        centered (bool): Whether black borders are placed so that the image is always centered.

    Returns:
        A tuple containing

        - **frame** (*np.array*): The image in letterbox format.
        - **border_px** (*np.array*): The border applied on each side [right, left, bottom, top] in pixels.
    """
    black = (0, 0, 0)
    (H, W, _) = image.shape
    if centered:
        sides = max(0, int(H - W)) // 2
        top_bot = max(0, int(W - H)) // 2
        border_frame = cv2.copyMakeBorder(image, top_bot, top_bot, sides, sides, cv2.BORDER_CONSTANT, black)
        return border_frame, np.array([sides, sides, top_bot, top_bot])
    else:
        right = max(0, int(H - W))
        bottom = max(0, int(W - H))
        border_frame = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, black)
        return border_frame, np.array([right, 0, bottom, 0])
