Detection
=====================

Welcome to the documentation about the detectors.

detector
---------------------------

.. automodule:: pytb.detection.detector

detector_factory
---------------------------

.. automodule:: pytb.detection.detector_factory

detection_manager
---------------------------

.. automodule:: pytb.detection.detection_manager

bboxes_2d_detector
---------------------------

.. toctree::

   bboxes_2d_detector

Hereby, you find two tables that (1) list the parameters in relation with the JSON config file that can be used for each detector and (2) give details on their usage.

.. list-table:: List of detectors and their parameters
   :widths: 50 50
   :header-rows: 1

   * - Model Type
     - Parameters
   * - BackgroundSubtractor (mean/median)
     - ``contour_thresh, intensity, max_last_images``
   * - BackgroundSubtractor (frame_diff)
     - ``contour_thresh, intensity``
   * - FasterRCNN / MaskRCNN
     - ``model_path*, input_width, input_height, use_coco_weights, GPU``
   * - YOLO4 (YOLOv 2, 3, 4)
     - ``model_path, config_path, input_width, input_height, conf_thresh, nms_thresh, nms_across_classes, GPU, haf_precision``
   * - YOLO5
     - ``model_path, input_width, input_height, conf_thresh, nms_thresh, nms_across_classes, GPU``
   * - Detectron2
     - ``model_path, config_path, conf_thresh, nms_thresh, GPU``

\* In FasterRCNN and MAskRCNN: model_path is not needed in case ``use_coco_weights`` is true.

.. list-table:: List of parameters and their usage
   :widths: 30 20 50
   :header-rows: 1

   * - Parameter
     - Default value
     - Description
   * - ``model_path``
     - /
     - The path to the weight of the model to be used (e.g. for YOLOv5, ending with .pt)
   * - ``config_path``
     - /
     - The path to the configuration file describing the model
   * - ``input_width`` / ``input_height``
     - 416
     - The input path of the image in the detector. This allows to setup the first layers of the network to match the image shape.
   * - ``conf_thresh``
     - 0
     - The minimum confidence threshold of the detected objects if the implementation allows to provide one.
   * - ``nms_thresh``
     - 0
     - The minimum non-max suppression threshold of the detected objects when the implementation allows to provide one.
   * - ``nms_across_classes``
     - true
     - Whether to perform the NMS algorithm across the different classes of object or separately.
   * - ``GPU``
     - false
     - Whether to use the GPU if available.
   * - ``haf_precision``
     - false
     - Whether to use the half precision capability of the recent GPU cards.
   * - ``use_coco_weights``
     - true
     - Whether to use the default weights available on PyTorch. If True, it will be downloading in cache automatically.
   * - ``contour_thresh``
     - 3
     - From cv2.approxPolyDP: Specifies the approximation accuracy. This is the maximum distance between the original curve and its approximation.
   * - ``intensity``
     - 50
     - The minimum intensity of the pixels in the foreground image.
   * - ``max_last_images``
     - 50
     - The number of last images that will be based to determine the brackground and the foreground (either with a mean or median computation)

.. automodule:: pytb.detection.bboxes.bboxes_2d_detector.bboxes_2d_detector