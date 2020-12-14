import cv2
import numpy as np
import timeit

class ImageHelper:

    @staticmethod
    def read_cv2(image_path):
        # TODO check is_file, extension, etc.
        return cv2.imread(image_path)

    @staticmethod
    def apply_roi_file(image, roi_path):
        roi = ImageHelper.read_cv2(roi_path)
        assert image.shape[:2] == roi.shape[:2], \
                "The mask image has not the same width or height as the frame to be masked."
        return cv2.bitwise_and(image, roi)

    @staticmethod
    def apply_roi_coords(image, roi_coords):
        roi = np.zeros(image.shape, dtype=np.uint8)
        polygon = np.array([roi_coords], dtype=np.int32)
        num_frame_channels = image.shape[2]
        mask_ignore_color = (255,) * num_frame_channels
        roi = cv2.fillPoly(roi, polygon, mask_ignore_color)
        return cv2.bitwise_and(image, roi)

    @staticmethod
    def add_borders(image, centered=False):
        """Add black border to 'frame' keep aspect ratio
        return the frame in letterbox format

        Args:
            image (np.ndarray): The image 

        Returns:
            np.ndarray: The image in letterbox format
        """
        black = (0, 0, 0)
        (H, W, _) = image.shape
        if centered:
            sides = max(0, int(H-W))//2
            top_bot = max(0,int(W-H))//2
            border_frame = cv2.copyMakeBorder(image, top_bot, top_bot, sides, sides, cv2.BORDER_CONSTANT, black)
        else:
            right = max(0, int(H-W))
            bottom = max(0,int(W-H))
            border_frame = cv2.copyMakeBorder(image, 0, bottom, 0, right, cv2.BORDER_CONSTANT, black)
        return border_frame
