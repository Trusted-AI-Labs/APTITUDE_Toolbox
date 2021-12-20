from typing import Tuple, Union
import logging
import numpy as np

from pytb.output.detection import Detection
import pytb.utils.image_helper as ih
import ast

log = logging.getLogger("aptitude-toolbox")


def pre_process(preprocess_parameters: dict, image: np.ndarray, prev_roi: np.ndarray = None,
                detection: Detection = None) \
        -> Tuple[np.ndarray, Union[np.ndarray, None]]:
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

    if "resize" in preprocess_parameters:
        resize_params = preprocess_parameters["resize"]
        image = ih.resize(image, resize_params["width"], resize_params["height"])
        log.debug("Image resized.")
        if detection is not None:
            detection.change_dims(resize_params["width"], resize_params["height"])
            log.debug("Detection resized.")

    if "border" in preprocess_parameters:
        border_params = preprocess_parameters["border"]
        image = ih.add_borders(image, centered=border_params.get("centered", False))
        log.debug("Borders added to the image.")
    return image, roi


def post_process(postprocess_parameters: dict, detection: Detection) -> Detection:
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
    if "resize_results" in postprocess_parameters:
        resize_res = postprocess_parameters["resize_results"]
        detection.change_dims(resize_res["width"], resize_res["height"])
        log.debug("Results were resized.")
    return detection
