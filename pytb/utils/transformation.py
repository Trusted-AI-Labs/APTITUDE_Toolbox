from typing import Tuple, Union
import logging
import numpy as np

from pytb.output.detection import Detection
import pytb.utils.image_helper as ih
import ast

log = logging.getLogger("aptitude-toolbox")


def pre_process(preprocess_parameters: dict, image: np.ndarray, prev_roi: np.ndarray = None,
                detection: Detection = None) \
        -> Tuple[np.ndarray, Union[np.ndarray, None], Union[np.ndarray, None], Union[Detection, None]]:
    if "roi" in preprocess_parameters:
        roi = prev_roi
        if prev_roi is None:
            roi_params = preprocess_parameters["roi"]
            # Apply a mask via a mask file
            if "path" in roi_params:
                roi = ih.get_roi_file(roi_params["path"])
                log.debug("ROI obtained from a mask file.")
            # Apply a mask via a polyline
            elif "coords" in roi_params:
                roi = ih.get_roi_coords(image, roi_params["coords"])
                log.debug("ROI obtained from polygon coordinates.")
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
                                 border_px[2]/ratio_height, border_px[3]/ratio_height], np.int)

    return image, roi, border_px, detection


def post_process(postprocess_parameters: dict,  detection: Detection) -> Detection:
    if "nms" in postprocess_parameters:
        nms_params = postprocess_parameters["nms"]
        detection.nms_filter(nms_params["pref_implem"], nms_params["nms_thresh"])
        log.debug("NMS algorithm applied.")
    if "coi" in postprocess_parameters:
        detection.class_filter(ast.literal_eval(postprocess_parameters["coi"]))
        log.debug("Only classes of interest were kept.")
    if "min_conf" in postprocess_parameters:
        detection.confidence_filter(postprocess_parameters["min_conf"])
        log.debug("Only detection reaching the confidence threshold were kept.")
    if "top_k" in postprocess_parameters:
        detection.top_k(postprocess_parameters["top_k"])
        log.debug("Only top K detections were kept.")
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
    if "resize_results" in postprocess_parameters:
        resize_res = postprocess_parameters["resize_results"]
        detection.change_dims(resize_res["width"], resize_res["height"])
        log.debug("Results were resized.")
    return detection
