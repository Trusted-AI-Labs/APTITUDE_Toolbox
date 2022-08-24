"""
Copyright (c) 2021-2022 UCLouvain, ICTEAM
Licensed under GPL-3.0 [see LICENSE for details]
Written by Jonathan Samelson (2021-2022)
"""

import ast
import logging

log = logging.getLogger("aptitude-toolbox")

valid_preproc_keys = ["border", "resize", "roi"]
valid_postproc_keys = ["coi", "nms", "min_conf", "max_height", "min_height",
                       "max_width", "min_width", "min_area", "top_k", "resize_results", "roi"]
valid_nms_values = ["cv2", "Malisiewicz"]

valid_proc_keys = ["task", "output_type", "model_type", "pref_implem", "params"]
valid_detector_type = ["YOLO4", "YOLO5", "MaskRCNN", "FasterRCNN", "BackgroundSubtractor", "Detectron2"]
valid_tracker_type = ["SORT", "DeepSORT", "Centroid", "IOU"]


def validate_preprocess_parameters(pre_params: dict) -> bool:
    """Check validity and compatibility between provided preprocess parameters.

    Args:
        pre_params (dict): the dictionary containing preprocess parameters.

    Returns:
        bool: whether it is a valid configuration.
    """
    if not pre_params:
        return True  # Empty dict for Preproc is valid

    list_keys = pre_params.keys()
    valid = True
    for key in list_keys:
        if key not in valid_preproc_keys:
            log.error("{} is not a valid entry for Preproc configuration.".format(key))
            valid = False

    if "border" in list_keys:
        if "centered" not in pre_params["border"]:
            log.error("\"border\" entry without \"centered\" sub-entry.")
            valid = False
        if not isinstance(pre_params["border"].get("centered"), bool):
            log.error("\"centered\" entry should be of type bool.")
    if "resize" in list_keys:
        if "width" not in pre_params["resize"]:
            log.error("\"resize\" entry without \"width\" sub-entries.")
            valid = False
        if "height" not in pre_params["resize"]:
            log.error("\"resize\" entry without \"height\" sub-entries.")
            valid = False
        if not isinstance(pre_params["resize"]["width"], int) \
                or not isinstance(pre_params["resize"]["height"], int):
            log.error("resize width or height must be of type int.")
            valid = False
        if pre_params["resize"]["width"] <= 0 or pre_params["resize"]["height"] <= 0:
            log.error("resize width or height must be positive.")
            valid = False
    if "roi" in list_keys:
        valid = _validate_roi_parameters(pre_params["roi"], preproc=True)
    return valid


def validate_postprocess_parameters(post_params: dict) -> bool:
    """Check validity and compatibility between provided postprocess parameters.

    Args:
        post_params (dict): the dictionary containing postprocess parameters.

    Returns:
        bool: whether it is a valid configuration.
    """
    if not post_params:
        return True  # Empty dict for postproc is valid

    list_keys = post_params.keys()
    valid = True
    for key in list_keys:
        if key not in valid_postproc_keys:
            log.error("{} is not a valid entry for postproc configuration.".format(key))
            valid = False

    if "coi" in list_keys:
        if not isinstance(post_params["coi"], str):
            log.error("\"coi\" entry must be of type str.")
            valid = False
        else:
            coi = ast.literal_eval(post_params["coi"])
            if not isinstance(coi, list):
                log.error("\"coi\" entry should evaluate to type list.")
                valid = False

    if "nms" in list_keys:
        if "pref_implem" not in post_params["nms"] or "nms_thresh" not in post_params["nms"]:
            log.error("\"nms\" entry without \"pref_implem\" or \"nms_tresh\" sub-entries")
            valid = False
        if post_params["nms"]["pref_implem"] not in valid_nms_values:
            log.error("\"pref_implem\" of nms unknown. Valid values are : {}".format(valid_nms_values))
            valid = False
        if post_params["nms"]["nms_thresh"] < 0 or post_params["nms"]["nms_thresh"] > 1:
            log.error("\"nms_thresh\" (postproc) value must be included between 0 and 1.")
            valid = False

    if "min_conf" in list_keys and (post_params["min_conf"] < 0 or post_params["min_conf"] > 1):
        log.error("\"min_conf\" (postproc) value must be included between 0 and 1.")
        valid = False
    if "max_height" in list_keys and (post_params["max_height"] < 0 or post_params["max_height"] > 1):
        log.error("\"max_height\" value must be included between 0 and 1.")
        valid = False
    if "min_height" in list_keys and (post_params["min_height"] < 0 or post_params["min_height"] > 1):
        log.error("\"min_height\" value must be included between 0 and 1.")
        valid = False
    if "max_width" in list_keys and (post_params["max_width"] < 0 or post_params["max_width"] > 1):
        log.error("\"max_width\" value must be included between 0 and 1.")
        valid = False
    if "min_width" in list_keys and (post_params["min_width"] < 0 or post_params["min_width"] > 1):
        log.error("\"min_width\" value must be included between 0 and 1.")
        valid = False
    if "min_area" in list_keys and post_params["min_area"] < 0:
        log.error("\"top_k\" value must be greater or equal to 0.")
        valid = False
    if "top_k" in list_keys and post_params["top_k"] < 0:
        log.error("\"top_k\" value must be greater or equal to 0.")
        valid = False

    if "roi" in list_keys:
        valid = _validate_roi_parameters(post_params["roi"], preproc=False)

    if "resize_results" in list_keys:
        if "width" not in post_params["resize_results"]:
            log.error("\"resize_results\" entry without \"width\" sub-entries.")
            valid = False
        if "height" not in post_params["resize_results"]:
            log.error("\"resize_results\" entry without \"height\" sub-entries.")
            valid = False
        if not isinstance(post_params["resize_results"]["width"], int) \
                or not isinstance(post_params["resize_results"]["height"], int):
            log.error("resize_results width or height must be of type int.")
            valid = False
        if post_params["resize_results"]["width"] <= 0 or post_params["resize_results"]["height"] <= 0:
            log.error("resize_results width or height must be positive.")
            valid = False

    return valid


def _validate_roi_parameters(roi_params: dict, preproc: bool) -> bool:
    valid = True
    if "path" not in roi_params and "coords" not in roi_params:
        log.error("\"roi\" entry without \"path\" or \"coords\" sub-entry.")
        valid = False
    if "path" in roi_params and not isinstance(roi_params["path"], str):
        log.error("\"path\" (ROI) entry must be of type str.")
        valid = False
    if "coords" in roi_params:
        if not isinstance(roi_params["coords"], str):
            log.error("\"coords\" entry must be of type str.")
            valid = False
        coords = ast.literal_eval(roi_params["coords"])
        if (not isinstance(coords, tuple)) or (not isinstance(coords[0], tuple)):
            log.error("\"coords\" entry should evaluate to type tuple of tuples.")
            valid = False
    if not preproc and "max_outside_roi_thresh" not in roi_params:
        log.error("\"max_outside_roi_thresh\" entry must be provided if in post proc parameters")
        valid = False
    if "max_outside_roi_thresh" in roi_params \
            and (roi_params["max_outside_roi_thresh"] < 0 or roi_params["max_outside_roi_thresh"] > 1):
        log.error("\"max_outside_roi_thresh\" must be included between 0 and 1")
        valid = False
    return valid


def validate_detector_parameters(det_params: dict) -> bool:
    """Check validity and compatibility between provided detector parameters.

    Args:
        det_params (dict): the dictionary containing detector parameters.

    Returns:
        bool: whether it is a valid configuration
    """
    list_keys = det_params.keys()
    valid = True

    for key in valid_proc_keys:
        if key not in list_keys:
            log.error("{} is missing in proc configuration.".format(key))
            valid = False

    if det_params["task"] != "detection":
        log.error("Detector parameters are being evaluated but \"task\" entry has not the value \"detection\"")
        valid = False

    if det_params["output_type"] != "bboxes2D":
        log.error("Invalid \"output_type\", \"bboxes2D\" is the only supported output at the moment")
        valid = False

    if det_params["model_type"] not in valid_detector_type:
        log.error("Invalid \"model_type\" in proc configuration.")
        valid = False

    if "pref_implem" not in det_params:
        log.error("\"pref_implem\" is missing proc configuration.")
        valid = False
    elif not isinstance(det_params["pref_implem"], str):
        log.error("The value of \"pref_implem\" sub-entry must be of type string.")
        valid = False

    if det_params["model_type"] == "YOLO4":
        valid = valid and _validate_yolo4_parameters(det_params)
    elif det_params["model_type"] == "YOLO5":
        valid = valid and _validate_yolo5_parameters(det_params)
    elif det_params["model_type"] == "Detectron2":
        valid = valid and _validate_detectron2_parameters(det_params)
    elif det_params["model_type"] == "MaskRCNN":
        valid = valid and _validate_maskrcnn_parameters(det_params)
    elif det_params["model_type"] == "FasterRCNN":
        valid = valid and _validate_fasterrcnn_parameters(det_params)
    elif det_params["model_type"] == "BackgroundSubtractor":
        valid = valid and _validate_backgroundsubtractor_parameters(det_params)
    else:
        log.error("Unknown detector \"model_type\": {}".format(det_params["model_type"]))
        valid = False

    return valid


def _validate_yolo4_parameters(det_params: dict) -> bool:
    valid = True
    yolo4_params = det_params["params"]

    if det_params["pref_implem"] not in ["cv2-DetectionModel", "cv2-ReadNet"]:
        log.error("Unknown implementation of YOLO4: {}".format(det_params["pref_implem"]))
        valid = False

    if "config_path" not in yolo4_params:
        log.error("\"config_path\" sub-entry is required in params for YOLO4 model type.")
        valid = False
    elif not isinstance(yolo4_params["config_path"], str):
        log.error("The value of \"config_path\" sub-entry must be of type string.")
        valid = False

    if "model_path" not in yolo4_params:
        log.error("\"model_path\" sub-entry is required in params for YOLO4 model type.")
        valid = False
    elif not isinstance(yolo4_params["model_path"], str):
        log.error("The value of \"model_path\" sub-entry must be of type string.")
        valid = False

    if "input_width" in yolo4_params \
            and not (isinstance(yolo4_params["input_width"], int) and yolo4_params["input_width"] > 0):
        log.error("\"input_width\" sub-entry must be of type int and must be positive.")
        valid = False
    if "input_height" in yolo4_params \
            and not (isinstance(yolo4_params["input_height"], int) and yolo4_params["input_height"] > 0):
        log.error("\"input_height\" sub-entry must be of type int and must be positive.")
        valid = False

    if "conf_thresh" in yolo4_params and (yolo4_params["conf_thresh"] < 0 or yolo4_params["conf_thresh"] > 1):
        log.error("\"conf_thresh\" (YOLO4 params) value must be included between 0 and 1.")
        valid = False
    if "nms_thresh" in yolo4_params and (yolo4_params["nms_thresh"] < 0 or yolo4_params["nms_thresh"] > 1):
        log.error("\"nms_thresh\" (YOLO4 params) value must be included between 0 and 1.")
        valid = False
    if "nms_across_classes" in yolo4_params and not isinstance(yolo4_params.get("nms_across_classes"), bool):
        log.error("\"nms_across_classes\" sub-entry must be of type bool.")
        valid = False
    if "GPU" in yolo4_params and not isinstance(yolo4_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    if "half_precision" in yolo4_params and not isinstance(yolo4_params.get("half_precision"), bool):
        log.error("\"half_precision\" sub-entry must be of type bool.")
        valid = False
    return valid


def _validate_yolo5_parameters(det_params: dict) -> bool:
    valid = True
    yolo5_params = det_params["params"]

    if det_params["pref_implem"] != "torch-Ultralytics":
        log.error("Unknown implementation of YOLO5: {}".format(det_params["pref_implem"]))
        valid = False

    if "model_path" not in yolo5_params:
        log.error("\"model_path\" sub-entry is required in params for YOLO5 model type.")
        valid = False
    elif not isinstance(yolo5_params["model_path"], str):
        log.error("The value of \"model_path\" sub-entry must be of type string.")
        valid = False

    if "input_width" in yolo5_params \
            and not (isinstance(yolo5_params["input_width"], int) and yolo5_params["input_width"] > 0):
        log.error("\"input_width\" sub-entry must be of type int and must be positive.")
        valid = False
    if "input_height" in yolo5_params \
            and not (isinstance(yolo5_params["input_height"], int) and yolo5_params["input_height"] > 0):
        log.error("\"input_height\" sub-entry must be of type int and must be positive.")
        valid = False

    if "conf_thresh" in yolo5_params and (yolo5_params["conf_thresh"] < 0 or yolo5_params["conf_thresh"] > 1):
        log.error("\"conf_thresh\" (YOLO5 params) value must be included between 0 and 1.")
        valid = False
    if "nms_thresh" in yolo5_params and (yolo5_params["nms_thresh"] < 0 or yolo5_params["nms_thresh"] > 1):
        log.error("\"nms_thresh\" (YOLO5 params) value must be included between 0 and 1.")
        valid = False
    if "nms_across_classes" in yolo5_params and not isinstance(yolo5_params.get("nms_across_classes"), bool):
        log.error("\"nms_across_classes\" sub-entry must be of type bool.")
        valid = False
    if "GPU" in yolo5_params and not isinstance(yolo5_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    return valid


def _validate_backgroundsubtractor_parameters(det_params: dict) -> bool:
    valid = True

    if det_params["pref_implem"] not in ["mean", "median", "frame_diff"]:
        log.error("Unknown implementation of BackgroundSubtractor: {}".format(det_params["pref_implem"]))
        valid = False

    bs_params = det_params["params"]
    if "contour_thresh" in bs_params and bs_params["contour_thresh"] < 0:
        log.error("\"contour_thresh\" (BackgroundSubtraction) value must be positive.")
        valid = False
    if "intensity" in bs_params and bs_params["intensity"] < 0:
        log.error("\"intensity\" value must be positive.")
        valid = False
    if "max_last_images" in bs_params and bs_params["max_last_images"] < 0:
        log.error("\"max_last_images\" value must be positive.")
        valid = False
    return valid


def _validate_detectron2_parameters(det_params: dict) -> bool:
    valid = True
    det2_params = det_params["params"]

    if det_params["pref_implem"] != "Default":
        log.error("Unknown implementation of Detectron2: {}".format(det_params["pref_implem"]))
        valid = False

    if "config_path" not in det2_params:
        log.error("\"config_path\" sub-entry is required in params for Detectron2 model type.")
        valid = False
    elif not isinstance(det2_params["config_path"], str):
        log.error("The value of \"config_path\" sub-entry must be of type string.")
        valid = False

    if "model_path" not in det2_params:
        log.error("\"model_path\" sub-entry is required in params for Detectron2 model type.")
        valid = False
    elif not isinstance(det2_params["model_path"], str):
        log.error("The value of \"model_path\" sub-entry must be of type string.")
        valid = False

    if "conf_thresh" in det2_params and (det2_params["conf_thresh"] < 0 or det2_params["conf_thresh"] > 1):
        log.error("\"conf_thresh\" (Detectron2) value must be included between 0 and 1.")
        valid = False
    if "nms_thresh" in det2_params and (det2_params["nms_thresh"] < 0 or det2_params["nms_thresh"] > 1):
        log.error("\"nms_thresh\" (Detectron2) value must be included between 0 and 1.")
        valid = False
    if "GPU" in det2_params and not isinstance(det2_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    return valid


def _validate_maskrcnn_parameters(det_params: dict) -> bool:
    valid = True

    mrcnn_params = det_params["params"]
    if det_params["pref_implem"] != "torch-resnet50":
        log.error("Unknown implementation of MaskRCNN: {}".format(det_params["pref_implem"]))
        valid = False

    if "input_width" in mrcnn_params \
            and not (isinstance(mrcnn_params["input_width"], int) and mrcnn_params["input_width"] > 0):
        log.error("\"input_width\" sub-entry must be of type int and must be positive.")
        valid = False
    if "input_height" in mrcnn_params \
            and not (isinstance(mrcnn_params["input_height"], int) and mrcnn_params["input_height"] > 0):
        log.error("\"input_height\" sub-entry must be of type int and must be positive.")
        valid = False

    if "GPU" in mrcnn_params and not isinstance(mrcnn_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in mrcnn_params and not isinstance(mrcnn_params.get("use_coco_weights"), bool):
        log.error("\"use_coco_weights\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in mrcnn_params and not mrcnn_params["use_coco_weights"] \
            and "model_path" not in mrcnn_params:
        log.error("If \"use_coco_weights\" is set to False, \"model_path\" must be an entry of params.")
        valid = False
    return valid


def _validate_fasterrcnn_parameters(det_params: dict) -> bool:
    valid = True

    fasterrcnn_params = det_params["params"]
    if det_params["pref_implem"] != "torch-resnet50":
        log.error("Unknown implementation of FasterRCNN: {}".format(det_params["pref_implem"]))
        valid = False

    if "input_width" in fasterrcnn_params \
            and not (isinstance(fasterrcnn_params["input_width"], int) and fasterrcnn_params["input_width"] > 0):
        log.error("\"input_width\" sub-entry must be of type int and must be positive.")
        valid = False
    if "input_height" in fasterrcnn_params \
            and not (isinstance(fasterrcnn_params["input_height"], int) and fasterrcnn_params["input_height"] > 0):
        log.error("\"input_height\" sub-entry must be of type int and must be positive.")
        valid = False

    if "GPU" in fasterrcnn_params and not isinstance(fasterrcnn_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in fasterrcnn_params and not isinstance(fasterrcnn_params.get("use_coco_weights"), bool):
        log.error("\"use_coco_weights\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in fasterrcnn_params and not fasterrcnn_params["use_coco_weights"] \
            and "model_path" not in fasterrcnn_params:
        log.error("If \"use_coco_weights\" is set to False, \"model_path\" must be an entry of params.")
        valid = False
    return valid


def validate_tracker_parameters(track_params: dict) -> bool:
    """Check validity and compatibility between provided detector parameters.

    Args:
        track_params (dict): the dictionary containing tracker parameters.

    Returns:
        bool: whether it is a valid configuration
    """
    list_keys = track_params.keys()
    valid = True

    for key in valid_proc_keys:
        if key not in list_keys:
            log.error("{} is missing in proc configuration.".format(key))
            valid = False

    if track_params["task"] != "tracking":
        log.error("Detector parameters are being evaluated but \"task\" entry has not the value \"tracking\"")
        valid = False

    if track_params["output_type"] != "bboxes2D":
        log.error("Invalid \"output_type\", \"bboxes2D\" is the only supported output at the moment")
        valid = False

    if track_params["model_type"] not in valid_tracker_type:
        log.error("Invalid \"model_type\" in proc configuration.")
        valid = False

    if "pref_implem" not in track_params:
        log.error("\"pref_implem\" is missing proc configuration.")
        valid = False
    elif not isinstance(track_params["pref_implem"], str):
        log.error("The value of \"pref_implem\" sub-entry must be of type string.")
        valid = False

    if track_params["model_type"] == "SORT":
        valid = valid and _validate_sort_parameters(track_params)
    elif track_params["model_type"] == "DeepSORT":
        valid = valid and _validate_deepsort_parameters(track_params)
    elif track_params["model_type"] == "Centroid":
        valid = valid and _validate_centroid_parameters(track_params)
    elif track_params["model_type"] == "IOU":
        valid = valid and _validate_iou_parameters(track_params)
    else:
        log.error("Unknown detector \"model_type\": {}".format(track_params["model_type"]))
        valid = False

    return valid


def _validate_sort_parameters(track_params: dict) -> bool:
    valid = True
    sort_params = track_params["params"]

    if track_params["pref_implem"] != "Abewley":
        log.error("Unknown implementation of SORT: {}".format(track_params["pref_implem"]))
        valid = False

    if "max_age" in sort_params and sort_params["max_age"] < 0:
        log.error("\"max_age\" value must be positive.")
        valid = False
    if "min_hits" in sort_params and sort_params["min_hits"] < 0:
        log.error("\"min_hits\" value must be positive.")
        valid = False
    if "iou_thresh" in sort_params and (sort_params["iou_thresh"] < 0 or sort_params["iou_thresh"] > 1):
        log.error("\"iou_thresh\" value must be included between 0 and 1.")
        valid = False
    if "memory_fade" in sort_params and sort_params["memory_fade"] < 1:
        log.error("\"memory_fade\" value must be greater than 1.")
        valid = False

    return valid


def _validate_deepsort_parameters(track_params: dict) -> bool:
    valid = True
    deepsort_params = track_params["params"]

    if track_params["pref_implem"] != "Leonlok":
        log.error("Unknown implementation of DeepSORT: {}".format(track_params["pref_implem"]))
        valid = False

    if "model_path" not in deepsort_params:
        log.error("\"model_path\" sub-entry is required in \"DeepSORT\" entry.")
        valid = False
    elif not isinstance(deepsort_params["model_path"], str):
        log.error("\"model_path\" (DeepSORT) must be of type string.")
        valid = False

    if "max_age" in deepsort_params and deepsort_params["max_age"] < 0:
        log.error("\"max_age\" value must be positive.")
        valid = False
    if "min_hits" in deepsort_params and deepsort_params["min_hits"] < 0:
        log.error("\"min_hits\" value must be positive.")
        valid = False
    if "iou_thresh" in deepsort_params and (deepsort_params["iou_thresh"] < 0 or deepsort_params["iou_thresh"] > 1):
        log.error("\"iou_thresh\" value must be included between 0 and 1.")
        valid = False
    if "max_cosine_dist" in deepsort_params \
            and (deepsort_params["max_cosine_dist"] < 0 or deepsort_params["max_cosine_dist"] > 1):
        log.error("\"max_cosine_dist\" value must be included between 0 and 1.")
        valid = False
    if "avg_det_conf" in deepsort_params:
        if not isinstance(deepsort_params.get("avg_det_conf"), bool):
            log.error("\"avg_det_conf\" must be of type bool.")
            valid = False
        if "avg_det_conf_thresh" not in deepsort_params:
            print("[ERROR \"avg_det_conf\" entry withtout \"avg_det_conf_thresh\" entry.")
            valid = False
        elif deepsort_params["avg_det_conf_thresh"] < 0 or deepsort_params["avg_det_conf_thresh"] > 1:
            log.error("\"avg_det_conf_thresh\" value must be included between 0 and 1.")
            valid = False
    if "most_common_class" in deepsort_params and not isinstance(deepsort_params.get("most_common_class"), bool):
        log.error("\"most_common_class\" must be of type bool.")
        valid = False

    return valid


def _validate_centroid_parameters(track_params: dict) -> bool:
    valid = True
    centroid_params = track_params["params"]

    if track_params["pref_implem"] != "Rosebrock":
        log.error("Unknown implementation of Centroid: {}".format(track_params["pref_implem"]))
        valid = False

    if "max_age" in centroid_params and centroid_params["max_age"] < 0:
        log.error("\"max_age\" value must be positive.")
        valid = False
    return valid


def _validate_iou_parameters(track_params: dict) -> bool:
    valid = True
    iou_params = track_params["params"]

    if track_params["pref_implem"] not in ["SimpleIOU", "KIOU"]:
        log.error("Unknown implementation of Centroid: {}".format(track_params["pref_implem"]))
        valid = False

    if "min_hits" in iou_params and iou_params["min_hits"] < 0:
        log.error("\"min_hits\" value must be positive.")
        valid = False
    if "iou_thresh" in iou_params and (iou_params["iou_thresh"] < 0 or iou_params["iou_thresh"] > 1):
        log.error("\"iou_thresh\" value must be included between 0 and 1.")
        valid = False
    return valid
