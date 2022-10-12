Tracking
=====================

Welcome to the documentation about the trackers.

tracker
---------------------------

.. automodule:: pytb.tracking.tracker

tracker_factory
---------------------------

.. automodule:: pytb.tracking.tracker_factory

tracking_manager
---------------------------

.. automodule:: pytb.tracking.tracking_manager

bboxes_2d_tracker
---------------------------

.. toctree::

   bboxes_2d_tracker

Hereby, you find two tables that (1) list the parameters in relation with the JSON config file that can be used for each tracker and (2) give details on their usage.

.. list-table:: List of trackers and their parameters
   :widths: 50 50
   :header-rows: 1

   * - Model Type
     - Parameters
   * - Centroid
     - ``max_age``
   * - IOU
     - ``min_hits, iou_thresh``
   * - KIOU
     - ``min_hits, iou_thresh, max_age``
   * - SORT
     - ``min_hits, iou_thresh, max_age, memory_fade``
   * - DeepSORT
     - ``model_path, min_hits, iou_thresh, max_age, memory_fade, max_cosine_dist, nn_budget, avg_det_conf, avg_det_conf_thresh, most_common_class``

.. list-table:: List of parameters and their usage
   :widths: 30 20 50
   :header-rows: 1

   * - Parameter
     - Default value
     - Description
   * - ``max_age``
     - 10
     - An object that is not tracked for max_age frame is removed from the memory.
   * - ``min_hits``
     - 3
     - Minimum of hits to start tracking the objects.
   * - ``iou_thresh``
     - 0.3 / 0.7*
     - The minimum IOU threshold to keep the association of a previously detected object.
   * - ``memory_fade``
     - 1.0
     - Above a value of 1.0, it enables a fading memory which gives less importance to the older tracks in the memory.
   * - ``model_path``
     - /
     - DeepSORT requires a model weight trained to track object features.
   * - ``max_cosine_dist``
     - 0.3
     - See underlying implementation.
   * - ``nn_budget``
     - null
     - See underlying implementation.
   * - ``avg_det_conf``
     - false
     - Whether the average detection confidence should be evaluated to filter out detection.
   * - ``avg_det_conf_thresh``
     - 0
     - If avg_det_conf is used, the thresholds defines the average confidence under which a detection will be filtered out.
   * - ``most_common_class``
     - false
     - If true, the object class will be the most common class detected over time.

\* In the case of DeepSORT, the default value of ``iou_thresh`` is 0.7, it showed better performance experimentally.


.. automodule:: pytb.tracking.bboxes.bboxes_2d_tracker.bboxes_2d_tracker
