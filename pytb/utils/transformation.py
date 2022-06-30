"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

from typing import Tuple, Optional
import logging
import numpy as np

from pytb.output.detection import Detection
from pytb.output.bboxes_2d import BBoxes2D
import pytb.utils.image_helper as ih
import ast

log = logging.getLogger("aptitude-toolbox")


def pre_process(preprocess_parameters: dict, image: np.ndarray, prev_roi: np.ndarray = None,
                detection: Detection = None) \
        -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[Detection]]:
    """Applies the preprocess parameters onto the image and optionally a detection.
    It consists of image transformation method (apply a region of interest, add a border, resize the image).

    Args:
        preprocess_parameters (dict): A dictionary containing the operations to be applied on the image.
            It consists of key-value pairs that should be validated against the validator module.
        image (np.ndarray): The image on which pre-process modification must be applied.
        prev_roi (np.ndarray): An image defining a binary mask to apply the ROI.
            This is optional and only needed to avoid repeated readings of the ROI file
            if the same ROI is used along a sequence of image.
        detection (Detection): If provided, returns the detection transformed to take into account
            the image modification (such as the modification of the dimensions).

    Returns:
        A tuple containing

        - **image** (*np.ndarray*): The modified detection, if provided as an input to take into account the image\
        modification.
        - **roi** (*Optional[np.ndarray]*): An optional frame of the ROI that was read, to be reused for next calls.
        - **border_px** (*Optional[np.ndarray]*): If "border" is to be applied, the number of pixels that was added\
         on each side of the frame in the following order: [right, left, bottom, top].
        - **detection** (*Optional[Detection]*): The detection transformed in accordance with the image modification.
    """
    if "roi" in preprocess_parameters:
        # Using previous ROI if exists to avoid repeated readings
        roi = prev_roi
        if prev_roi is None:
            roi = _get_roi(preprocess_parameters["roi"], image.shape)
        image = ih.apply_roi(image, roi)
    else:
        roi = None

    border_px = None
    if "border" in preprocess_parameters:
        if detection is not None:
            detection.change_dims(image.shape[1], image.shape[0])
            log.debug("Detection resized to match image size")
        border_params = preprocess_parameters["border"]
        image, border_px = ih.add_borders(image, centered=border_params.get("centered", False))
        log.debug("Borders added to the image.")
        if detection is not None:
            detection.add_borders(border_px)
            log.debug("Borders added to the detections")

    if "resize" in preprocess_parameters:
        prev_dims = image.shape
        resize_params = preprocess_parameters["resize"]
        image = ih.resize(image, resize_params["width"], resize_params["height"])
        log.debug("Image resized.")
        if detection is not None:
            detection.change_dims(resize_params["width"], resize_params["height"])
            log.debug("Detection resized.")

        if border_px is not None:
            new_dims = image.shape
            ratio_width = prev_dims[1] / new_dims[1]
            ratio_height = prev_dims[0] / new_dims[0]
            border_px = np.array([border_px[0]/ratio_width, border_px[1]/ratio_width,
                                 border_px[2]/ratio_height, border_px[3]/ratio_height], np.uint8)

    return image, roi, border_px, detection


def post_process(postprocess_parameters: dict,  detection: Detection, prev_roi: np.ndarray = None) \
        -> Tuple[Detection, Optional[np.ndarray]]:
    """Applies the postprocess parameters onto the detection. It mainly consists of filtering method
    that removes a set of bounding boxes based on a set of thresholds.
    Learn more about those methods in the output classes (e.g. BBoxes2D) where those methods are implemented.
    In this function, the methods are called in a specific order that should provide the best results
    (yet, it is not guaranteed and one could change the order to obtain better results as the order matters).

    Args:
        postprocess_parameters (dict): A dictionary containing the operations to be applied on the detection.
            It consists of key-value pairs that should be validated against the validator module.
        detection (Detection): The detection on which post-process operations must be applied.
            The operations may vary depending on the type of the detection
            (only BBoxes2D & BBoxes2DTrack are supported at the moment).
        prev_roi (np.ndarray): An image defining a binary mask to apply the ROI.
            This is optional and only needed to avoid repeated readings of the ROI file
            if the same ROI is used along a sequence of image.

    Returns:
        A tuple containing

        - **detection** (*Detection*): The modified detection, after applying the post-process operation.
        - **roi** (*Optional[np.ndarray]*): An optional frame of the ROI that was read, to be reused for next calls.
    """
    if isinstance(detection, BBoxes2D) and detection.number_objects > 0:
        # Using previous ROI if exists to avoid repeated readings
        roi = prev_roi

        # Order of the below operations matters
        if "coi" in postprocess_parameters:
            detection.class_filter(ast.literal_eval(postprocess_parameters["coi"]))
            log.debug("Only classes of interest were kept.")
        if "min_conf" in postprocess_parameters:
            detection.confidence_filter(postprocess_parameters["min_conf"])
            log.debug("Only detection reaching the confidence threshold were kept.")
        if "max_height" in postprocess_parameters:
            detection.height_filter(postprocess_parameters["max_height"], max_filter=True)
            log.debug("Only detections below max height threshold were kept.")
        if "min_height" in postprocess_parameters:
            detection.height_filter(postprocess_parameters["min_height"], max_filter=False)
            log.debug("Only detections above min height threshold were kept.")
        if "max_width" in postprocess_parameters:
            detection.width_filter(postprocess_parameters["max_width"], max_filter=True)
            log.debug("Only detections below max width threshold were kept.")
        if "min_width" in postprocess_parameters:
            detection.width_filter(postprocess_parameters["min_width"], max_filter=False)
            log.debug("Only detections above max width threshold were kept.")
        if "min_area" in postprocess_parameters:
            detection.min_area_filter(postprocess_parameters["min_area"])
            log.debug("Only detections above min area threshold were kept.")
        # borders_detection comes from preprocess
        if "borders_detection" in postprocess_parameters:
            border_px = postprocess_parameters["borders_detection"]
            detection.remove_borders(border_px)
            log.debug("Results were adjusted to take borders into account")
        if "roi" in postprocess_parameters:
            roi_params = postprocess_parameters["roi"]
            if prev_roi is None:
                roi = _get_roi(roi_params, (detection.dim_height, detection.dim_width))
            detection.roi_filter(roi, roi_params["max_outside_roi_thresh"])
            log.debug("Only classes in the ROI were kept.")
        if "nms" in postprocess_parameters:
            nms_params = postprocess_parameters["nms"]
            detection.nms_filter(nms_params["pref_implem"], nms_params["nms_thresh"])
            log.debug("NMS algorithm applied.")
        if "top_k" in postprocess_parameters:
            detection.top_k(postprocess_parameters["top_k"])
            log.debug("Only top K detections were kept.")
        if "resize_results" in postprocess_parameters:
            resize_res = postprocess_parameters["resize_results"]
            detection.change_dims(resize_res["width"], resize_res["height"])
            log.debug("Results were resized.")
        return detection, roi


def _get_roi(roi_params: dict, image_shape: tuple) -> np.ndarray:
    """
    Args:
        roi_params (dict): The parameters to apply the region of interest (ROI)
            as part of the preproc or postproc parameters.
        image_shape (tuple): The shape of the image for which it should be resized if the parameter "path" is chosen.
            Otherwise, in case of a polygon coords, it is assumed it is provided in the correct dimensions.

    Returns:
        np.ndarray: The binary mask of the ROI, either from the polygon coords or the image path.
    """
    # Apply a mask via a mask file
    if "path" in roi_params:
        roi = ih.get_roi_file(roi_params["path"])
        roi = ih.resize(roi, image_shape[1], image_shape[0])
        log.debug("ROI obtained from a mask file.")
        return roi
    # Apply a mask via a polyline, the coords should be provided in the desired dimensions
    elif "coords" in roi_params:
        roi = ih.get_roi_coords(image_shape, roi_params["coords"])
        log.debug("ROI obtained from polygon coordinates.")
        return roi
