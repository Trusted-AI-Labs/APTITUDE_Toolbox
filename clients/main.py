import tracker_client_images as tci
import tracker_client_video as tcv

import argparse
import os
import logging
import coloredlogs

if __name__ == "__main__":
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", required=True,
                    help="path to detector config (json file)")
    ap.add_argument("-t", "--tracker", required=True,
                    help="path to tracker config (json file)")
    ap.add_argument("-c", "--classes", required=True,
                    help="path to classes (json file)")
    ap.add_argument("-p", "--path", type=str, required=True,
                    help="path to video file / folder path containing images (.png/.jpg/.bmp) in lexical order")
    ap.add_argument("-fi", "--frame_interval", type=int, default=1,
                    help="interval between two detections + tracking. Default is 1")
    ap.add_argument("-gt", "--ground_truth_path", type=str, default=None,
                    help="path to ground truth file in MOT format.")
    ap.add_argument("-hl", "--headless", action='store_true',
                    help="whether the video is shown as it processed")
    ap.add_argument("-rp", "--record_path", type=str, default=None,
                    help="path of the output video file")
    ap.add_argument("-rf", "--record_fps", type=int, default=10,
                    help="fps of the output video file")
    ap.add_argument("-mp", "--mot_path", type=str, default=None,
                    help="path to the result of tracking in MOT format")
    ap.add_argument("-a", "--async", action='store_true',
                    help="for video file only. whether video reading is asynchronous")
    ap.add_argument("-l", "--log_level", type=str, default="info",
                    help="Log level."
                         "Possible values : \"NOTSET\", \"DEBUG\", \"INFO\", \"WARNING\", \"ERROR\", \"CRITICAL\".")
    args = vars(ap.parse_args())

    assert args["log_level"] in ["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], \
        "Invalid log level. Possible values : \"NOTSET\", \"DEBUG\", \"INFO\", \"WARNING\", \"ERROR\", \"CRITICAL\"."

    log = logging.getLogger("aptitude-toolbox")
    coloredlogs.install(level=args["log_level"],
                        fmt="%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s")

    if os.path.isdir(args["path"]):
        tci.main(args["detector"], args["tracker"], args["classes"],
                 args["path"], args["frame_interval"], args["record_path"], args["record_fps"],
                 args["mot_path"], args["headless"], args["ground_truth_path"],)

    else:
        tcv.main(args["detector"], args["tracker"], args["classes"],
                 args["path"], args["frame_interval"], args["record_path"], args["record_fps"],
                 args["mot_path"], args["headless"], args["async"], args["ground_truth_path"])
