from typing import Tuple, Union

import numpy as np

from pytb.output.detection import Detection
import pytb.utils.image_helper as ih
import ast

def pre_process(preprocess_parameters: dict, image: np.ndarray, prev_roi: np.ndarray = None) \
        -> Tuple[np.ndarray, Union[np.ndarray, None]]:

    if "roi" in preprocess_parameters:
        roi = prev_roi
        if prev_roi is None:
            roi_params = preprocess_parameters["roi"]
            # Apply a mask via a mask file
            if "path" in roi_params:
                roi = ih.get_roi_file(roi_params["path"])
            # Apply a mask via a polyline
            elif "coords" in roi_params:
                roi = ih.get_roi_coords(image, roi_params["coords"])
        image = ih.apply_roi(image, roi)
    else:
        roi = None

    if "resize" in preprocess_parameters:
        resize_params = preprocess_parameters["resize"]
        image = ih.resize(image, resize_params["width"], resize_params["height"])

    if "border" in preprocess_parameters:
        border_params = preprocess_parameters["border"]
        image = ih.add_borders(image, centered=border_params.get("centered", False))
    return image, roi


def post_process(postprocess_parameters: dict, detection: Detection) -> Detection:
    if "nms" in postprocess_parameters:
        nms_params = postprocess_parameters["nms"]
        detection.nms_filter(nms_params["pref_implem"], nms_params["nms_thresh"])
    if "coi" in postprocess_parameters:
        detection.class_filter(ast.literal_eval(postprocess_parameters["coi"]))
    if "min_conf" in postprocess_parameters:
        detection.confidence_filter(postprocess_parameters["min_conf"])
    if "top_k" in postprocess_parameters:
        detection.top_k(postprocess_parameters["top_k"])
    if "max_height" in postprocess_parameters:
        detection.height_filter(postprocess_parameters["max_height"], max_filter=True)
    if "min_height" in postprocess_parameters:
        detection.height_filter(postprocess_parameters["min_height"], max_filter=False)
    if "max_width" in postprocess_parameters:
        detection.width_filter(postprocess_parameters["max_width"], max_filter=True)
    if "min_width" in postprocess_parameters:
        detection.width_filter(postprocess_parameters["min_width"], max_filter=False)
    if "min_area" in postprocess_parameters:
        detection.min_area_filter(postprocess_parameters["min_area"])
    if "resize_results" in postprocess_parameters:
        resize_res = postprocess_parameters["resize_results"]
        detection.change_dims(resize_res["width"], resize_res["height"])
    return detection
