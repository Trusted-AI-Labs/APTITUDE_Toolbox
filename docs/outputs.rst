Outputs & Postproc
=====================

Welcome to the documentation about the outputs.

All functions are defined in details further down the page.

detection
""""""""""""""""

.. automodule:: pytb.output.detection

bboxes_2d
"""""""""""""""""

Herebelow, you can find a table of postproc parameters in relation with the JSON config file. They are used to filter out the detection in the case of bounding boxes 2D.
**When a parameter is not specified, the filter is simply not applied to the set of detections.**

.. list-table:: List of postproc parameters and their usage
   :widths: 20 20 60
   :header-rows: 1

   * - Parameter
     - Sub-parameter
     - Description
   * - ``nms``
     - ``pref_implem`` (str)
     -  Either ``"cv2"`` or ``"Malisiewicz"``. Apply a Non-Max Suppression algorithm. Based on a threshold, it removes bounding boxes that overlap and only keeps one entity based on a selection criteria. This criteria depends on the implementation.
   * -
     - ``nms_thresh`` (float)
     - The threshold to apply the Non-Max Suppression ranging from 0 to 1.
   * - ``min_conf`` (float)
     -
     - Keeps only the detections that have a confidence score above the threshold.
   * - ``max_height`` (int)
     -
     - Keeps only the detections whose the height is below the provided percentage of the frame.
   * - ``min_height`` (int)
     -
     - Keeps only the detections whose the height is above the provided percentage of the frame.
   * - ``max_width`` (int)
     -
     - Keeps only the detections whose the width is below the provided percentage of the frame.
   * - ``min_width`` (int)
     -
     - Keeps only the detections whose the width is above the provided percentage of the frame.
   * - ``min_area`` (int)
     -
     - Keeps only the detections whose the area is above the minimum area threshold. More specifically, if the detection's width multiplied by the detection's height is not (strictly) above the threshold, it is filtered out.
   * - ``top_k`` (int)
     -
     - Keeps only the K most confident prediction. If a choice has to be made between two detections that have the same confidence score, this choice is arbitrary.
   * - ``coi`` (str)
     -
     - Format is a string of a list of int (e.g. "[0, 2]"). Keeps only the detections that are included in the set of classes of interest.
   * - ``roi``
     - ``path`` (str)
     - The path to a binary mask where white pixels represent the Region of Interest (ROI) and the black pixels represent the regions to be ignored.
   * -
     - ``max_outside_roi_thresh`` (float)
     - The threshold of area percentage above which a bounding box has to be removed ranging from 0 to 1.
   * - ``resize_results``
     - ``width``, ``height`` (int)
     - Scales up/down the dimensions of the bounding boxes in accordance to the provided ``width`` and ``height``.


.. automodule:: pytb.output.bboxes_2d

bboxes_2d_track
""""""""""""""""

.. automodule:: pytb.output.bboxes_2d_track

