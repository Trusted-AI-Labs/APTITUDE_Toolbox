import tracker_client_images as tci
import tracker_client_video as tcv

import argparse
import os

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--detector", required=True,
                    help="path to detector config (json file)")
    ap.add_argument("-t", "--tracker", required=True,
                    help="path to tracker config (json file)")
    ap.add_argument("-c", "--classes", required=True,
                    help="path to classes (json file)")
    ap.add_argument("-p", "--path", type=str, required=True,
                    help="path to video file / folder path")
    ap.add_argument("-fi", "--frame_interval", type=int, default=1,
                    help="interval between two detections + tracking. Default is 1")
    ap.add_argument("-gt", "--ground_truth_path", type=str, default=None,
                    help="path to ground truth file in MOT format. For image sequence, CSV format (read more in docs)")
    ap.add_argument("-hl", "--headless", action='store_true',
                    help="whether the video is shown as it processed")
    ap.add_argument("-rp", "--record_path", type=str, default=None,
                    help="path of the output video file")
    ap.add_argument("-rf", "--record_fps", type=int, default=10,
                    help="fps of the output video file")
    ap.add_argument("-mp", "--mot_path", type=str, default=None,
                    help="path to the result of tracking in mot format.")
    ap.add_argument("-a", "--async", action='store_true',
                    help="for video file only. whether video reading is async")
    args = vars(ap.parse_args())

    if os.path.isdir(args["path"]):
        tci.main(args["detector"], args["tracker"], args["classes"],
                 args["path"], args["frame_interval"], args["record_path"], args["record_fps"],
                 args["headless"], args["ground_truth_path"])

    else:
        tcv.main(args["detector"], args["tracker"], args["classes"],
                 args["path"], args["frame_interval"], args["record_path"], args["record_fps"],
                 args["mot_path"], args["headless"], args["async"], args["ground_truth_path"])
