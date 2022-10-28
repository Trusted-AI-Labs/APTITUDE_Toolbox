Utils & Preproc
=====================

Welcome to the documentation about the utilities.

Hereby, you find one table that list the preprocess parameters in relation with the JSON config file. They are used to transform the image before applying the detector or the tracker.

All functions are defined in details further down the page.

**When a parameter is not specified, the transformation is simply not applied to the image.**

.. list-table:: List of preproc parameters and their usage
   :widths: 20 20 120
   :header-rows: 1

   * - Parameter
     - Sub-parameter
     - Description
   * - ``border``
     - ``centered`` (bool)
     -  Whether black borders are placed so that the image is always centered.
   * - ``resize``
     - ``width``, ``height`` (int)
     - The width/height of the image after resizing, in pixels.
   * - ``roi*``
     - ``coords`` (str)
     - The set of the polygon coords that defines the Region of Interest (the white pixels). It must be of the following format: ``"(0, 0), (450, 0), (450, 200), (0, 200)"``
   * -
     - ``path`` (str)
     - The path to a binary mask where white pixels represent the Region of Interest (ROI) and the black pixels represent the regions to be ignored.

\* For the Region of Interest (ROI), either choose ``coords`` or ``path``, not both!

**NB:** Note that applying a ROI to the image could lead to weird results if the model has never seen the mask before.
If you wish to apply a ROI, it is advised to apply it as a post-process parameters (see dedicated section),
as it will suppress objects outside the ROI without altering the image.

image_helper
"""""""""""""""""

.. automodule:: pytb.utils.image_helper

transformation
"""""""""""""""""

.. automodule:: pytb.utils.transformation

util
""""""""""""""""

.. automodule:: pytb.utils.util

validator
""""""""""""""""

.. automodule:: pytb.utils.validator

video_capture_async
"""""""""""""""""""""

.. automodule:: pytb.utils.video_capture_async

