#!/usr/bin/env python
# coding: utf-8
"""
Driver code for RPL Tag Camera Engine.
"""

import argparse
import json
from datetime import datetime
from typing import Optional

import cv2
from rpl_tag_engine_driver.utils import camera_utils, measurement_utils, object_utils


class RPLTagEngine:
    """
    A class to handle the detection and pose estimation of RPLTags in a video feed.

    Attributes:
        db_filename (str): Path to the RPLTag database.
        camera_fname (str): Path to the camera calibration file.
        verbose (bool): Flag for enabling verbose output.
        vid (cv2.VideoCapture): Video capture object.
        width (float): Width of the video frame.
        height (float): Height of the video frame.
        the_db (object_utils.RPLTagDatabase): Loaded RPLTag database.
        the_camera (dict): Loaded camera parameters.
        camera_params (Tuple): Camera parameters extracted from the calibration matrix.
        tag_size (float): Size of the AprilTag in meters.
        tag_family (str): Tag family used for detection.
    """

    def __init__(self, database: str, camera: str, verbose: bool = False):
        """
        Initialize the RPLTagEngine with database and camera calibration files.
        """
        self.db_filename = database
        self.camera_fname = camera
        self.verbose = verbose
        self.status = "INIT"
        self.vid = cv2.VideoCapture(0)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.the_db = object_utils.RPLTagDatabase()
        self.the_db.load(self.db_filename)
        self.the_camera = camera_utils.load_camera(self.camera_fname)
        self.camera_params = camera_utils.camera_params_from_matrix(self.the_camera["newcameramtx"])
        self.tag_size = 1.0
        self.tag_family = "tag36h11"

    def grab_image(self) -> None:
        """
        Capture an image from the video stream and save it with a timestamp.
        """
        ret, self.img = self.vid.read()
        if not ret:
            raise RuntimeError("Failed to capture image from camera.")

        self.datestamp = datetime.now().strftime("%Y%b%d_%H%M%S")

    def analyze_image(self, img: Optional[cv2.Mat] = None) -> None:
        """
        Analyze the image to detect tags, estimate poses, and link detections to registered multitags.

        Args:
            img (Optional[cv2.Mat]): Image to analyze. If None, uses the last captured image.
        """
        if img is None:
            img = self.img

        self.img_fname = "temp.jpg"
        cv2.imwrite(self.img_fname, img)

        # Detect individual tags
        self.und_fname = "undistorted.jpg"
        self.ovl_fname = "annotated.jpg"

        self.these_tag_IDs, self.these_corners, self.ippe_results = (
            measurement_utils.wrapped_up_detection_pipeline_function(
                img,
                self.the_camera,
                tag_size=self.tag_size,
                tag_family=self.tag_family,
                undistorted_fname=self.und_fname,
                overlay_fname=self.ovl_fname,
            )
        )

        if self.verbose:
            print(f"these_tag_IDs: {len(self.these_tag_IDs)}")
            print(f"these_corners: {len(self.these_corners)}")
            print(f"ippe_results: {len(self.ippe_results)}")

        # Sort into proximate groups
        self.tag_groups = measurement_utils.wrapped_up_multitag_finder(self.these_corners)
        print(f"tag_groups: {len(self.tag_groups)}")

        # Link detections to registered multitags
        self.legal_multitags = measurement_utils.find_multitag_candidates(
            self.tag_groups, self.these_tag_IDs, self.the_db, verbose=False
        )
        if self.verbose:
            print(f"legal_multitags: {len(self.legal_multitags)}")

        # Estimate poses of detected multitags
        self.mt_poses = measurement_utils.detected_multitag_poses(
            self.legal_multitags, self.camera_params, self.the_db, self.these_corners, self.ippe_results, self.tag_size
        )
        if self.verbose:
            print(f"mt_poses: {len(self.mt_poses)}")

        # Package mt_poses as a dictionary
        self.detections = {
            self.the_db.multitags.entries[p[0]].name: [p[3][0].tolist(), p[3][1].tolist()] for p in self.mt_poses
        }

    def print_detections(self) -> None:
        """
        Print the detected multitags and their poses.
        """
        for multitag_name, (rvec, tvec) in self.detections.items():
            rstr = "[ " + " ".join([f"{x:6.3f}" for x in rvec]) + " ]"
            tstr = "[ " + " ".join([f"{x:6.2f}" for x in tvec]) + " ] inches"
            print(f"{multitag_name:20} {rstr} {tstr}")

    def cleanup(self) -> None:
        """
        Release resources and close any open windows.
        """
        self.vid.release()
        cv2.destroyAllWindows()

    def on_demand_script(self) -> str:
        """
        Run the RPLTagEngine and return serialized detection results as a JSON string.
        """
        self.status = "BUSY"
        self.grab_image()
        self.analyze_image()

        serialized_detections = json.dumps(self.detections)
        self.cleanup()
        self.status = "IDLE"
        return serialized_detections


def take_it_for_a_spin() -> None:
    """
    Run a test of the RPLTagEngine with command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run RPLTagEngine with the provided database and camera files.")
    parser.add_argument("--database", default="RPLtag.db", help="Database filename")
    parser.add_argument("--camera", required=True, help="Camera calibration filename or directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    rpltag_reader = RPLTagEngine(args.database, args.camera, args.verbose)

    rpltag_reader.grab_image()
    rpltag_reader.analyze_image()
    rpltag_reader.print_detections()
    rpltag_reader.cleanup()


if __name__ == "__main__":
    # uncomment this line for quick one pass command line invocation
    # take_it_for_a_spin()

    # uncomment this line for use as use in FastAPI execute script
    pass
