import ast
import logging

log = logging.getLogger("aptitude-toolbox")

valid_preproc_keys = ["border", "resize", "roi"]
valid_postproc_keys = ["coi", "nms", "min_conf", "max_height", "min_height",
                       "max_width", "min_width", "min_area", "top_k", "resize_results", "roi"]
valid_nms_values = ["cv2", "Malisiewicz"]
valid_detector_keys = ["Detector", "BBoxes2DDetector", "YOLO", "MRCNN", "FASTERRCNN", "BackgroundSubtractor",
                       "Detectron2"]
valid_tracker_keys = ["Tracker", "BBoxes2DTracker", "SORT", "DeepSORT", "Centroid", "IOU"]


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
        return True  # Empty dict for Postproc is valid

    list_keys = post_params.keys()
    valid = True
    for key in list_keys:
        if key not in valid_postproc_keys:
            log.error("{} is not a valid entry for Postproc configuration.".format(key))
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
            log.error("\"nms_thresh\" (Postproc) value must be included between 0 and 1.")
            valid = False

    if "min_conf" in list_keys and (post_params["min_conf"] < 0 or post_params["min_conf"] > 1):
        log.error("\"min_conf\" (Postproc) value must be included between 0 and 1.")
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
    """Check validity and compatibility between provided tracker parameters.

    Args:
        det_params (dict): the dictionary containing detector parameters.

    Returns:
        bool: whether it is a valid configuration
    """
    list_keys = det_params.keys()
    valid = True
    for key in list_keys:
        if key not in valid_detector_keys:
            log.error("{} is not a valid entry for Proc configuration.".format(key))
            valid = False

    if "Detector" not in list_keys:
        log.error("\"Detector\" entry is missing in Proc Configuration.")
        valid = False

    if "type" not in det_params["Detector"]:
        log.error("\"type\" sub-entry is required in \"Detector\" entry.")
        valid = False
    elif det_params["Detector"]["type"] == "BBoxes2DDetector":
        valid = valid and _validate_bboxes2ddetector_parameters(det_params)
    else:
        log.error("Detector type {} is unknown.".format(det_params["Detector"]["type"]))
        valid = False

    return valid


def _validate_bboxes2ddetector_parameters(det_params: dict) -> bool:
    b2d_params = det_params["BBoxes2DDetector"]
    valid = True
    if "model_type" not in b2d_params:
        log.error("\"model_type\" sub-entry is required in \"BBoxes2DDetector\" entry.")
        valid = False
    elif not isinstance(b2d_params["model_type"], str):
        log.error("The value of \"model_type\" sub-entry must be of type string.")
        valid = False
    elif b2d_params["model_type"] == "YOLO":
        valid = valid and _validate_yolo_parameters(det_params)
    elif b2d_params["model_type"] == "BackgroundSubtractor":
        valid = valid and _validate_backgroundsubtraction_parameters(det_params)
    elif b2d_params["model_type"] == "Detectron2":
        valid = valid and _validate_detectron2_parameters(det_params)
    elif b2d_params["model_type"] == "MRCNN":
        valid = valid and _validate_mrcnn_parameters(det_params)
    elif b2d_params["model_type"] == "FASTERRCNN":
        valid = valid and _validate_fasterrcnn_parameters(det_params)
    else:
        log.error("The model type (Detector) {} is unknown.".format(b2d_params["model_type"]))
        valid = False

    if "pref_implem" not in b2d_params:
        log.error("\"pref_implem\" sub-entry is required in \"BBoxes2DDetector\" entry.")
        valid = False
    elif not isinstance(b2d_params["pref_implem"], str):
        log.error("The value of \"pref_implem\" sub-entry must be of type string.")
        valid = False

    # CV2 and DefaultPredictor of Detectron2 needs config
    needs_config = b2d_params["pref_implem"] in ["cv2-DetectionModel", "cv2-ReadNet", "Default"]

    # All implementations except BackgroundSubtractor and {MRCNN,FASTERRCNN} needs models
    # {MRCNN,FASTERRCNN} can download a model dynamically.
    needs_model = b2d_params["model_type"] not in ["BackgroundSubtractor", "MRCNN", "FASTERRCNN"]

    if "config_path" not in b2d_params and needs_config:
        log.error("\"config_path\" sub-entry is required in \"BBoxes2DDetector\" entry.")
        valid = False
    elif needs_config and not isinstance(b2d_params["config_path"], str):
        log.error("The value of \"config_path\" sub-entry must be of type string")
        valid = False

    if "model_path" not in b2d_params and needs_model:
        log.error("\"model_path\" sub-entry is required in \"BBoxes2DDetector\" entry.")
        valid = False
    elif needs_model and not isinstance(b2d_params["model_path"], str):
        log.error("The value of \"model_path\" sub-entry must be of type string.")
        valid = False

    if "input_width" in b2d_params \
            and not isinstance(b2d_params["input_width"], int) \
            and b2d_params["input_width"] > 0:
        log.error("\"input_width\" sub-entry must be of type int and must be positive.")
        valid = False
    if "input_height" in b2d_params \
            and not isinstance(b2d_params["input_height"], int) \
            and b2d_params["input_height"] > 0:
        log.error("\"input_height\" sub-entry must be of type int and must be positive.")
        valid = False
    return valid


def _validate_yolo_parameters(det_params: dict) -> bool:
    yolo_params = det_params["YOLO"]
    if not yolo_params:
        return True  # Empty dict for YOLO is valid
    valid = True
    if "conf_thresh" in yolo_params and (yolo_params["conf_thresh"] < 0 or yolo_params["conf_thresh"] > 1):
        log.error("\"conf_thresh\" (YOLO) value must be included between 0 and 1.")
        valid = False
    if "nms_thresh" in yolo_params and (yolo_params["nms_thresh"] < 0 or yolo_params["nms_thresh"] > 1):
        log.error("\"nms_thresh\" (YOLO) value must be included between 0 and 1.")
        valid = False
    if "nms_across_classes" in yolo_params and not isinstance(yolo_params.get("nms_across_classes"), bool):
        log.error("\"nms_across_classes\" sub-entry must be of type bool.")
        valid = False
    if "GPU" in yolo_params and not isinstance(yolo_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    if "half_precision" in yolo_params and not isinstance(yolo_params.get("half_precision"), bool):
        log.error("\"half_precision\" sub-entry must be of type bool.")
        valid = False
    return valid


def _validate_backgroundsubtraction_parameters(det_params: dict) -> bool:
    bs_params = det_params["BackgroundSubtractor"]
    if not bs_params:
        return True  # Empty dict for BS is valid
    valid = True
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
    det2_params = det_params["Detectron2"]
    if not det2_params:
        return True  # Empty dict for Detectron2 is valid
    valid = True
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


def _validate_mrcnn_parameters(det_params: dict) -> bool:
    mrcnn_params = det_params["MRCNN"]
    if not mrcnn_params:
        return True  # Empty dict for MRCNN is valid
    valid = True
    if "GPU" in mrcnn_params and not isinstance(mrcnn_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in mrcnn_params and not isinstance(mrcnn_params.get("use_coco_weights"), bool):
        log.error("\"use_coco_weights\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in mrcnn_params and not mrcnn_params["use_coco_weights"] \
            and "model_path" not in det_params["BBoxes2DDetector"]:
        log.error("If \"use_coco_weights\" is set to False, \"model_path\" must be provided in BBoxes2DDetector.")
        valid = False
    return valid


def _validate_fasterrcnn_parameters(det_params: dict) -> bool:
    fasterrcnn_params = det_params["FASTERRCNN"]
    if not fasterrcnn_params:
        return True  # Empty dict for FASTERRCNN is valid
    valid = True
    if "GPU" in fasterrcnn_params and not isinstance(fasterrcnn_params.get("GPU"), bool):
        log.error("\"GPU\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in fasterrcnn_params and not isinstance(fasterrcnn_params.get("use_coco_weights"), bool):
        log.error("\"use_coco_weights\" sub-entry must be of type bool.")
        valid = False
    if "use_coco_weights" in fasterrcnn_params and not fasterrcnn_params["use_coco_weights"] \
            and "model_path" not in det_params["BBoxes2DDetector"]:
        log.error("If \"use_coco_weights\" is set to False, \"model_path\" must be provided in BBoxes2DDetector.")
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
    for key in list_keys:
        if key not in valid_tracker_keys:
            log.error("{} is not a valid entry for Proc configuration.".format(key))
            valid = False

    if "Tracker" not in list_keys:
        log.error("\"Detector\" entry is missing in Proc Configuration.")
        valid = False

    if "type" not in track_params["Tracker"]:
        log.error("\"type\" sub-entry is required in \"Detector\" entry.")
        valid = False
    elif track_params["Tracker"]["type"] == "BBoxes2DTracker":
        valid = valid and _validate_bboxes2dtracker_parameters(track_params)
    else:
        log.error("Tracker type {} is unknown.".format(track_params["Detector"]["type"]))
        valid = False

    return valid


def _validate_bboxes2dtracker_parameters(track_params: dict) -> bool:
    b2t_params = track_params["BBoxes2DTracker"]
    valid = True
    if "model_type" not in b2t_params:
        log.error("\"model_type\" sub-entry is required in \"BBoxes2DTracker\" entry.")
        valid = False
    elif not isinstance(b2t_params["model_type"], str):
        log.error("The value of \"model_type\" sub-entry must be of type string.")
        valid = False
    elif b2t_params["model_type"] == "SORT":
        valid = valid and _validate_sort_parameters(track_params)
    elif b2t_params["model_type"] == "DeepSORT":
        valid = valid and _validate_deepsort_parameters(track_params)
    elif b2t_params["model_type"] == "Centroid":
        valid = valid and _validate_centroid_parameters(track_params)
    elif b2t_params["model_type"] == "IOU":
        valid = valid and _validate_iou_parameters(track_params)
    else:
        log.error("The model type (Tracker) {} is unknown.".format(b2t_params["model_type"]))
        valid = False

    if "pref_implem" not in b2t_params:
        log.error("\"pref_implem\" sub-entry is required in \"BBoxes2DTracker\" entry.")
        valid = False
    elif not isinstance(b2t_params["pref_implem"], str):
        log.error("The value of \"pref_implem\" sub-entry must be of type string.")
        valid = False

    return valid


def _validate_sort_parameters(track_params: dict) -> bool:
    sort_params = track_params["SORT"]
    if not sort_params:
        return True  # Empty dict for SORT is valid
    valid = True
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
    deepsort_params = track_params["DeepSORT"]
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
    centroid_params = track_params["Centroid"]
    if not centroid_params:
        return True  # Empty dict for Centroid is valid
    if "max_age" in centroid_params and centroid_params["max_age"] < 0:
        log.error("\"max_age\" value must be positive.")
        return False
    return True


def _validate_iou_parameters(track_params: dict) -> bool:
    iou_params = track_params["IOU"]
    if not iou_params:
        return True
    valid = True
    if "min_hits" in iou_params and iou_params["min_hits"] < 0:
        log.error("\"min_hits\" value must be positive.")
        valid = False
    if "iou_thresh" in iou_params and (iou_params["iou_thresh"] < 0 or iou_params["iou_thresh"] > 1):
        log.error("\"iou_thresh\" value must be included between 0 and 1.")
        valid = False
    return valid
