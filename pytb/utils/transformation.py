import numpy as np

from pytb.output.detection import Detection
import pytb.utils.image_helper as ih


def pre_process(preprocess_parameters: dict, image: np.ndarray) -> np.ndarray:
    if "roi" in preprocess_parameters:
        roi_params = preprocess_parameters["roi"]
        # Apply a mask via a mask file
        if "path" in roi_params:
            image = ih.apply_roi_file(image, roi_params["path"])
        # Apply a mask via a polyline
        elif "coords":
            image = ih.apply_roi_coords(image, roi_params["coords"])

    if "resize" in preprocess_parameters:
        resize_params = preprocess_parameters["resize"]
        image = ih.resize(image, resize_params["width"], resize_params["height"])

    if "border" in preprocess_parameters:
        border_params = preprocess_parameters["border"]
        image = ih.add_borders(image, centered=border_params["centered"])
    return image

def post_process(postprocess_parameters: dict, detection: Detection) -> Detection:
    if "nms" in postprocess_parameters:
        nms_params = postprocess_parameters["nms"]
        detection.nms_filter(nms_params["pref_implem"], nms_params["nms_thresh"])
    if "coi" in postprocess_parameters:
        detection.class_filter(postprocess_parameters["coi"])
    if "min_conf" in postprocess_parameters:
        detection.confidence_filter(postprocess_parameters["min_conf"])
    if "top_k" in postprocess_parameters:
        detection.top_k(postprocess_parameters["top_k"])
    return detection