import cv2
import numpy as np
import urllib.request
import ast
import logging

log = logging.getLogger("aptitude-toolbox")

try:
    from turbojpeg import TurboJPEG

    tjpeg = TurboJPEG()
except:
    tjpeg = None
    log.warning("TurboJPEG could not be found, using cv2 to decode images instead.")


def get_cv2_img_from_str(path: str, flags=cv2.IMREAD_COLOR):
    with open(path, 'rb') as buffer:
        if tjpeg is not None and (path.endswith(".jpg") or path.endswith(".jpeg")):
            return tjpeg.decode(buffer.read())
        else:
            nparr = np.frombuffer(buffer.read(), dtype=np.uint8)
            return cv2.imdecode(nparr, flags)


def get_cv2_img_from_buffer(buffer, flags=cv2.IMREAD_COLOR):
    nparr = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv2.imdecode(nparr, flags)


def get_cv2_img_from_url(url, flags=cv2.IMREAD_COLOR):
    req = urllib.request.Request(url)
    return get_cv2_img_from_buffer(urllib.request.urlopen(req), flags)


def resize(image: np.ndarray, width: int, height: int):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def get_roi_file(roi_path: str):
    return get_cv2_img_from_str(roi_path, flags=cv2.IMREAD_COLOR)


def get_roi_coords(image_shape: tuple, roi_coords: str):
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


def apply_roi(image, roi):
    assert image.shape[:2] == roi.shape[:2], \
        "The mask image has not the same width or height as the frame to be masked."
    return cv2.bitwise_and(image, roi)


def add_borders(image: np.ndarray, centered=False):
    """Add black border to 'frame' keep aspect ratio
    return the frame in letterbox format and the number of black pixels on each side

    Args:
        image (np.ndarray): The image
        centered (bool): Whether black borders are placed so that the image is always centered

    Returns:
        np.ndarray: The image in letterbox format
        np.array: The border applied on each side (right, left, bottom, top) in pixels
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
