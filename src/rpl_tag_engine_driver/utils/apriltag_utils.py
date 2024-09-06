"""
`apriltag_utils.py`

This module provides utility functions for working with AprilTag detections. It includes functions for retrieving and converting detection data, visualizing detection results, and performing pose estimation.

Key Functions:
- `value_by_field`: Extracts a specific field value from an AprilTag detection.
- `dict_from_detection`: Converts an AprilTag detection to a dictionary of field names and values.
- `sketch_detections`: Plots the centers of detected AprilTags on a 2D plot.
- `print_detection_essentials`: Prints essential details from AprilTag detections, including ID, center, rotation, and translation vectors.
- `tag_opoints`: Generates 3D coordinates of the corners of an AprilTag based on its size.
- `ippe`: Solves the Iterative Pose Pose Estimation (IPPE) method for pose reconstruction.
- `build_rstr` and `build_tstr`: Formats rotation and translation vectors as strings.
- `print_ippe_solutions`: Prints the results of IPPE solutions including rotation vectors, translation vectors, and errors.
- `apriltag_image`: Detects AprilTags in static images and optionally saves or displays the annotated results.
- `detect_apriltags`: Detects AprilTags in a single image file.
- `destroy_cv_windows`: Closes all OpenCV windows and waits briefly for proper closure.

This module facilitates the detection, visualization, and analysis of AprilTags in images, making it easier to integrate AprilTag functionality into Python applications.
"""

import os
from argparse import ArgumentParser

import apriltag
import cv2
import matplotlib.pyplot as plt
import numpy as np


def value_by_field(detection: apriltag.Detection, field):
    """
    Retrieve a value from a detection by its field name.

    Args:
        detection (Detection): The detection object from which to retrieve the value.
        field (str): The field name to retrieve.

    Returns:
        The value corresponding to the field name if it exists, otherwise None.
    """
    fields = detection._print_fields
    if field not in fields:
        return None
    return detection[fields.index(field)]


def dict_from_detection(detection: apriltag.Detection):
    """
    Create a dictionary from a detection's fields and values.

    Args:
        detection (Detection): The detection object to convert.

    Returns:
        dict: A dictionary where keys are field names and values are field values.
    """
    detection_dict = {}
    for field, value in zip(detection._print_fields, detection):
        detection_dict[field] = value
    return detection_dict


def sketch_detections(detections):
    """
    Plot the centers of detections on a 2D plot.

    Args:
        detections (list): A list of detections.
    """
    for detection in detections[0::4]:
        ID = value_by_field(detection, "ID")
        center = value_by_field(detection, "Center")
        if center is not None:
            plt.plot(center[0], center[1], "ko")
            plt.text(center[0], center[1], str(ID), fontsize=12)

    plt.gca().invert_yaxis()
    plt.show()


def print_detection_essentials(detections):
    """
    Print essential information from detections including ID, center, rotation vector, and translation vector.

    Args:
        detections (list): A list of detections.
    """
    for index, detection in enumerate(detections[0::4]):
        ID = value_by_field(detection, "ID")
        center = value_by_field(detection, "Center")
        pose = np.array(detections[4 * index + 1])
        rvec, _ = cv2.Rodrigues(pose[:3, :3])
        rvec = rvec.T[0]
        tvec = pose[:3, 3]

        center_str = "[" + "".join([f"{v:7.2f}" for v in center]) + "]"
        rvec_str = "[" + "".join([f"{v:6.2f}" for v in rvec]) + "]"
        tvec_str = "[" + "".join([f"{v:6.2f}" for v in tvec]) + "]"

        print(f"{index:2}: {ID:2} {center_str} {rvec_str} {tvec_str}")


def tag_opoints(tag_size):
    """
    Generate the 3D coordinates of the corners of a tag.

    Args:
        tag_size (float): The size of the tag.

    Returns:
        numpy.ndarray: A 3D array of the tag's corner points.
    """
    opoints = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]).reshape(-1, 1, 3) * 0.5 * tag_size
    return opoints


def ippe(opoints, ipoints, camera_matrix, dist_coeffs=None):
    """
    Solve the IPPE (Iterative Pose Pose Estimation) method for pose reconstruction.

    Args:
        opoints (numpy.ndarray): The 3D object points.
        ipoints (numpy.ndarray): The 2D image points.
        camera_matrix (numpy.ndarray): The camera matrix.
        dist_coeffs (numpy.ndarray, optional): The distortion coefficients.

    Returns:
        tuple: A tuple containing the result, rotation vectors, translation vectors, and errors.
    """
    pts2d_i_np = np.ascontiguousarray(ipoints).reshape((-1, 1, 2))  # solvePnP needs contiguous arrays
    pts3d_i_np = np.ascontiguousarray(opoints).reshape((-1, 3))

    return cv2.solvePnPGeneric(
        objectPoints=pts3d_i_np,
        imagePoints=pts2d_i_np,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE,
    )


def build_rstr(rvec):
    """
    Build a formatted string for the rotation vector.

    Args:
        rvec (numpy.ndarray): The rotation vector.

    Returns:
        str: A formatted string representing the rotation vector.
    """
    return "[" + "".join([f"{v:7.3f}" for v in rvec]) + "]"


def build_tstr(tvec):
    """
    Build a formatted string for the translation vector.

    Args:
        tvec (numpy.ndarray): The translation vector.

    Returns:
        str: A formatted string representing the translation vector.
    """
    return "[" + "".join([f"{v:8.3f}" for v in tvec]) + "]"


def print_ippe_solutions(rvecs, tvecs, errs):
    """
    Print the IPPE solutions including rotation vectors, translation vectors, and errors.

    Args:
        rvecs (list): A list of rotation vectors.
        tvecs (list): A list of translation vectors.
        errs (list): A list of errors associated with each solution.
    """
    print("          rvec                     tvec               err ")
    print("----------------------- --------------------------    ----")
    for rvec, tvec, e in zip(rvecs, tvecs, errs):
        rstr = build_rstr(rvec.T[0])
        tstr = build_tstr(tvec.T[0])
        estr = f"{e[0]:7.2f}"
        print(rstr, tstr, estr)


def apriltag_image(
    input_images=["AprilTag/media/input/single_tag.jpg", "AprilTag/media/input/multiple_tags.jpg"],
    camera_params=(3156.71852, 3129.52243, 359.097908, 239.736909),
    tag_size=1.0,  # inches
    output_images=False,
    display_images=True,
    detection_window_name="AprilTag",
    verbose=True,
):
    """
    Detect AprilTags from static images and optionally save or display the results.

    Args:
        input_images (list of str): List of image file paths to process.
        camera_params (tuple): Camera parameters (fx, fy, cx, cy).
        tag_size (float): The size of the AprilTag in inches.
        output_images (bool): Flag to save images annotated with detections.
        display_images (bool): Flag to display images annotated with detections.
        detection_window_name (str): Title of the window displaying the detection results.
        verbose (bool): Flag for verbose output in detection.

    Returns:
        tuple: Result and overlay images.
    """

    parser = ArgumentParser(description="Detect AprilTags from static images.")
    apriltag.add_arguments(parser)
    options = None

    verbose_detect_tags = 3 if verbose else 0

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    results = []
    overlays = []

    for image_path in input_images:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Unable to read image {image_path}.")
            continue

        print(f"Reading {os.path.split(image_path)[1]}...\n")

        result, overlay = apriltag.detect_tags(
            img,
            detector,
            camera_params=camera_params,
            tag_size=tag_size,
            vizualization=3,
            verbose=verbose_detect_tags,
            annotation=True,
        )

        if output_images:
            output_path = os.path.join(
                "AprilTag/media/output", os.path.basename(image_path).replace(os.path.splitext(image_path)[1], ".jpg")
            )
            cv2.imwrite(output_path, overlay)

        if display_images:
            cv2.imshow(detection_window_name, overlay)
            while cv2.waitKey(5) < 0:
                pass

        results.append(result)
        overlays.append(overlay)

    return results, overlays


def detect_apriltags(
    input_image,
    camera_params=(3156.71852, 3129.52243, 359.097908, 239.736909),
    tag_size=1.0,  # inches
    verbose=True,
):
    """
    Detect AprilTags from a single image.

    Args:
        input_image (str): Path to the image file.
        camera_params (tuple): Camera parameters (fx, fy, cx, cy).
        tag_size (float): The size of the AprilTag in inches.
        verbose (bool): Flag for verbose output in detection.

    Returns:
        tuple: Result of the detection.
    """
    parser = ArgumentParser(description="Detect AprilTags from static images.")
    apriltag.add_arguments(parser)
    options = None

    verbose_detect_tags = 3 if verbose else 0

    detector = apriltag.Detector(options, searchpath=apriltag._get_dll_path())

    img = cv2.imread(input_image)
    if img is None:
        print(f"Error: Unable to read image {input_image}.")
        return None

    result, _ = apriltag.detect_tags(
        img,
        detector,
        camera_params=camera_params,
        tag_size=tag_size,
        visualization=0,
        verbose=verbose_detect_tags,
        annotation=False,
    )

    return result


def destroy_cv_windows():
    """
    Close all OpenCV windows and wait briefly to ensure proper closure.
    """
    cv2.destroyAllWindows()
    cv2.waitKey(1)
