"""
This module provides utilities for measuring and processing AprilTag poses,
handling multitag detections, and interfacing with RPLTag databases.
"""

import apriltag_utils
import camera_utils
import cv2
import multitag_utils
import numpy as np
import object_utils


def add_measurements(
    measure_these: list[int],
    mt_poses: list[list],
    RPLTag_DB: object_utils.RPLTagDatabase,
    img_fname: str,
    camera_fname: str,
    verbose: bool = False,
    camera_only: bool = True,
) -> None:
    """
    Compute and add relative poses between a set of reference tags and detected tags.

    Args:
        measure_these (list of int): List of RPLTag IDs to measure.
        mt_poses (list of list): List of detected RPLTag poses.
        RPLTag_DB (object_utils.RPLTagDatabase): In-memory RPLTag database.
        img_fname (str): Filename of the image used for provenance.
        camera_fname (str): Filename of the camera calibration file.
        verbose (bool): If True, prints additional information. Default is False.
        camera_only (bool): If True, only records the pose of the reference tag. Default is True.

    Returns:
        None
    """
    if verbose:
        print(" " * 4, end=" ")
        for m_ref in measure_these:
            print(f"{m_ref:7}", end=" ")
        print()

    meas_added = 0
    out_of = 0

    for m_ref in measure_these:
        print(f"{m_ref:3}:", end=" ")
        r_ref = search_multitag_records(m_ref, mt_poses)
        if not r_ref:
            print()
            continue

        c_ref = -1
        record = [img_fname, camera_fname, r_ref[3][0], r_ref[3][1], c_ref, r_ref]
        result = RPLTag_DB.new_measurement(c_ref, m_ref, record)

        if camera_only:
            break

        for m_other in measure_these:
            if m_other > m_ref:
                r_other = search_multitag_records(m_other, mt_poses)
                if not r_other:
                    print(" " * 7, end=" ")
                    continue

                relative_rvec, relative_tvec = relative_pose(r_ref[3], r_other[3])
                record = [img_fname, camera_fname, relative_rvec, relative_tvec, r_ref, r_other]
                result = RPLTag_DB.new_measurement(m_ref, m_other, record)
                if result:
                    meas_added += 1
                out_of += 1

                if verbose:
                    print(f"{np.linalg.norm(relative_tvec):7.2f}", end=" ")
            else:
                if verbose:
                    print(" " * 7, end=" ")
        if verbose:
            print()
    if verbose:
        print(f"MEASUREMENTS added: {meas_added} out of {out_of} computed.")


def relative_pose(
    ref_pose_vecs: tuple[np.ndarray, np.ndarray], other_pose_vecs: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the relative pose between two sets of pose vectors.

    Args:
        ref_pose_vecs (tuple): A tuple containing the reference rotation vector and translation vector.
        other_pose_vecs (tuple): A tuple containing the other rotation vector and translation vector.

    Returns:
        tuple: Relative rotation vector and translation vector.
    """
    rot_ref, _ = cv2.Rodrigues(ref_pose_vecs[0])
    rot_oth, _ = cv2.Rodrigues(other_pose_vecs[0])
    new_rvec, _ = cv2.Rodrigues(np.matmul(rot_ref.T, rot_oth))
    new_tvec = other_pose_vecs[1] - ref_pose_vecs[1]
    return new_rvec, new_tvec


def search_multitag_records(this_mt: int, mt_pose_records: list[list]) -> list:
    """
    Search for multitag records with a specific ID.

    Args:
        this_mt (int): The ID of the multitag to search for.
        mt_pose_records (list of list): List of multitag pose records.

    Returns:
        list: The record for the specified multitag ID, or an empty list if not found.
    """
    for r in mt_pose_records:
        if r[0] == this_mt:
            return r
    return []


def order_legal_multitag_tags(
    mt_ID: int, detection_ID_pairs: list[list], tagdb: object_utils.RPLTagDatabase
) -> list[list]:
    """
    Orders detected tag ID pairs to match the tag template in the database.

    Args:
        mt_ID (int): The ID of the multitag to use as the template.
        detection_ID_pairs (list of list): List of detected tag ID pairs.
        tagdb (object_utils.RPLTagDatabase): Database containing multitag definitions.

    Returns:
        list: Ordered list of detection ID pairs matching the multitag template.
    """
    this_multitag = tagdb.multitags.entries[mt_ID]
    ordered_tag_pairs = []

    for tag_ID in this_multitag.tags:
        for pair in detection_ID_pairs:
            if pair[1] == tag_ID:
                ordered_tag_pairs.append(pair)
                break

    if len(ordered_tag_pairs) != len(detection_ID_pairs):
        print("FAIL - did not match all tag_IDs: detected", detection_ID_pairs, "required", this_multitag.tags)

    return ordered_tag_pairs


def detected_legal_multitag_list(
    legal_multitags: dict[int, dict[int, list[list]]], tagdb: object_utils.RPLTagDatabase
) -> list[list]:
    """
    Converts and sorts detected multitag pairs into a list matching the multitag templates.

    Args:
        legal_multitags (dict): Dictionary with multitag IDs and their detected pairs.
        tagdb (object_utils.RPLTagDatabase): Database containing multitag definitions.

    Returns:
        list: List of detected multitag entries with ordered tag pairs.
    """
    legal_multitag_list = []

    for mt_ID in legal_multitags:
        for g in legal_multitags[mt_ID]:
            ordered = order_legal_multitag_tags(mt_ID, legal_multitags[mt_ID][g], tagdb)
            legal_multitag_list.append([mt_ID, g, ordered])

    return legal_multitag_list


def tag_corners_from_these_corners(
    detection_index_list: list[int], these_corners: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Retrieves corners corresponding to detected indices.

    Args:
        detection_index_list (list): List of detection indices or pairs of indices.
        these_corners (list): List of all detected corners.

    Returns:
        list: List of corners corresponding to the detection indices.
    """
    tag_corners = []

    for d in detection_index_list:
        if isinstance(d, int):
            tag_corners.append(these_corners[d])
        elif isinstance(d, list):
            tag_corners.append(these_corners[d[0]])

    return tag_corners


def detected_multitag_poses(
    found_multitag_candidates: dict,
    camera_params: dict,
    tagdb: object_utils.RPLTagDatabase,
    these_corners: list,
    ippe_results: list,
    tag_size: float = 1.0,
    verbose: bool = False,
) -> list:
    """
    Computes poses for detected multitag candidates based on camera parameters and IPPE results.

    Args:
        found_multitag_candidates (dict): Dictionary of detected multitag candidates.
        camera_params (dict): Camera parameters for pose estimation.
        tagdb (object_utils.RPLTagDatabase): Database containing multitag definitions.
        these_corners (list): List of detected corners in the image.
        ippe_results (list): IPPE results for single tag poses.
        tag_size (float): Size of the tag. Default is 1.0.
        verbose (bool): If True, prints additional debug information. Default is False.

    Returns:
        list: List of multitag poses with estimated rotation and translation vectors.
    """
    multitag_list = detected_legal_multitag_list(found_multitag_candidates, tagdb)

    for mt_dex, mt_record in enumerate(multitag_list):
        mt_ID = mt_record[0]
        tag_group = mt_record[1]
        ordered_tag_list = mt_record[2]

        if verbose:
            print(f"{mt_ID:2} {tag_group:2}   ", ordered_tag_list)

        Ntags = len(ordered_tag_list)
        if Ntags > 1:
            tag_corners = tag_corners_from_these_corners(ordered_tag_list, these_corners)
            rvecs, tvecs = multitag_utils.consistent_coplanar_poses(
                tag_corners, tag_size, camera_params, kit_and_kaboodle=False
            )

            if verbose:
                print("ordered_tag_list", ordered_tag_list)
                print("rvecs", rvecs)
                print("tvecs", tvecs)

            multitag_list[mt_dex].append([rvecs[0], tvecs[0]])

        elif Ntags == 1:
            errs = ippe_results[ordered_tag_list[0][0]][3]
            if multitag_utils.pose_rotation_ambiguity_test(errs):
                rvecs = ippe_results[ordered_tag_list[0][0]][1]
                tvecs = ippe_results[ordered_tag_list[0][0]][2]
                which_one = 0 if errs[1] >= errs[0] else 1
                multitag_list[mt_dex].append(
                    [rvecs[which_one].flatten(), tvecs[which_one].flatten()]
                )  # *** THIS IS EXPEDIENT HACK
                if verbose:
                    print("rvecs", rvecs[which_one].flatten())
                    print("tvecs", tvecs[which_one].flatten())
            else:
                # Single tag, two solutions too close to choose between -- discard this tag as "un-estimable"
                multitag_list[mt_dex].append([])
                if verbose:
                    contrast = np.abs(errs[0] - errs[1]) / (errs[0] + errs[1])
                    print(
                        f"DISCARD THIS POSE ESTIMATE: e0 = {errs[0]:0.3f}, e1 = {errs[1]:0.3f}, contrast = {contrast:0.2f}"
                    )

    return multitag_list


def wrapped_up_detection_pipeline_function(
    img: np.ndarray,
    camera: dict,
    tag_size: float = 1.0,
    tag_family: str = "tag36h11",
    undistorted_fname: str = "undistorted.png",
    overlay_fname: str = "overlay.png",
) -> tuple:
    """
    Processes an image through a complete AprilTag detection pipeline, including undistortion,
    tag detection, and IPPE pose estimation.

    Args:
        img (np.ndarray): The input image to process.
        camera (dict): Camera calibration data including matrix, distortion coefficients, and other parameters.
        tag_size (float): The size of the AprilTag in real-world units. Default is 1.0.
        tag_family (str): The family of AprilTag used. Default is 'tag36h11'.
        undistorted_fname (str): Filename for saving the undistorted image. Default is 'undistorted.png'.
        overlay_fname (str): Filename for saving the image with tag detections overlayed. Default is 'overlay.png'.

    Returns:
        tuple: A tuple containing:
            - these_tag_IDs (list): List of detected tag IDs.
            - these_corners (list): List of detected tag corners.
            - ippe_results (list): Results of the IPPE (Iterative Pose and Pose Estimation) for the detected tags.
    """
    # Extract camera parameters from the camera calibration data
    mtx = camera["mtx"]
    dist = camera["dist"]
    rvecs = camera["rvecs"]
    tvecs = camera["tvecs"]
    img_h = camera["img_h"]
    img_w = camera["img_w"]
    newcameramtx = camera["newcameramtx"]
    roi = camera["roi"]
    print(rvecs, tvecs, img_h, img_w, roi)
    camera_params = camera_utils.camera_params_from_matrix(newcameramtx)

    # Undistort the input image
    undistorted_image = camera_utils.undistort(img, mtx, dist, newcameramtx)
    cv2.imwrite(undistorted_fname, undistorted_image)

    # Detect AprilTags and generate annotated image
    detections, overlay = apriltag_utils.apriltag_image(
        input_images=[undistorted_fname],
        camera_params=camera_params,
        tag_size=tag_size,
        output_images=False,
        display_images=False,
        detection_window_name="AprilTag",
        verbose=False,
    )
    cv2.imwrite(overlay_fname, overlay)

    # Extract detected tag IDs and corners
    these_tag_IDs = multitag_utils.value_from_all_tags(detections, "ID")
    these_corners = multitag_utils.value_from_all_tags(detections, "Corners")

    # Compute IPPE results
    ippe_results = multitag_utils.many_tag_ippe(these_corners, tag_size, newcameramtx, distCoeffs=None)

    return these_tag_IDs, these_corners, ippe_results


def wrapped_up_multitag_finder(these_corners: list) -> list:
    """
    Identifies and groups detected tags into multitag clusters based on their spatial proximity.

    Args:
        these_corners (list): List of detected tag corners.

    Returns:
        list: List of sets, each containing indices of tags grouped together.
    """
    if not these_corners:
        return []

    these_centers = np.array([np.mean(cs, axis=0) for cs in these_corners])

    tag_neighbors = multitag_utils.find_tag_neighbors(these_corners)
    tag_clusters = multitag_utils.aggregate_tag_clusters(tag_neighbors)

    all_detection_numbers = set(range(len(these_centers)))
    tags_in_clusters = set.union(*tag_clusters) if tag_clusters else set()
    not_in_clusters = all_detection_numbers - tags_in_clusters

    tag_groupings = []
    for m_tag in tag_clusters:
        tag_groupings.append(m_tag)
    for s_tag in not_in_clusters:
        tag_groupings.append({s_tag})

    return tag_groupings


def best_multitag_matches(membership_candidates: dict, verbose: bool = False) -> list:
    """
    Determines the best multitag matches from a list of candidate multitag groupings.

    Args:
        membership_candidates (dict): Dictionary where keys are tag indices and values are lists of candidate multitag matches.
        verbose (bool): If True, prints detailed output. Default is False.

    Returns:
        list: List of best matches for each detected tag.
    """

    def decide_which_mt(candidates: list) -> list:
        best_matches = [[0, 0]]
        for mt in candidates:
            if mt[1] == best_matches[0][1]:
                best_matches.append(mt)
            if mt[1] > best_matches[0][1]:
                best_matches = [mt.copy()]
        return best_matches

    best_candidates = []
    for t in membership_candidates:
        best = decide_which_mt(membership_candidates[t])
        best_candidates.append([t, best])
        if verbose:
            print(f"{t:3} {best[0][0]:3} {best[0][1]:3}")
    return best_candidates


def find_multitag_candidates(
    tag_groups: list, these_tag_IDs: list, RPLtag_db: object_utils.RPLTagDatabase, verbose: bool = False
) -> dict:
    """
    Finds potential multitag candidates from detected tag groupings and registered multitag database.

    Args:
        tag_groups (list): List of tag clusters identified by detection index.
        these_tag_IDs (list): List of detected tag IDs.
        RPLtag_db: Database object containing registered multitags.
        verbose (bool): If True, prints detailed output. Default is False.

    Returns:
        dict: Dictionary of potential multitag candidates with tags aggregated by multitag ID and group index.
    """
    membership_candidates = {}
    if verbose:
        print(" G   T   ID     MT tags      GROUP tags               MT_in_GROUP  MISSING")

    for g_dex, t_group in enumerate(tag_groups):
        for t_dex in t_group:
            tag_ID = these_tag_IDs[t_dex]
            if tag_ID in RPLtag_db.multitags.membership:
                possible_hits = RPLtag_db.multitags.membership[tag_ID]
                for mt_ID in possible_hits:
                    if mt_ID not in membership_candidates:
                        membership_candidates[mt_ID] = dict()
                    if g_dex not in membership_candidates[mt_ID]:
                        membership_candidates[mt_ID][g_dex] = []
                    membership_candidates[mt_ID][g_dex].append([t_dex, tag_ID])
                    if verbose:
                        print(f"{g_dex:2} {t_dex:2} {tag_ID:2} {mt_ID:2}")

    # Remove incomplete or incompatible multitag candidates
    keys_to_drop = []
    for k in membership_candidates:
        tID_list = RPLtag_db.multitags.entries[k].tags
        for g in membership_candidates[k]:
            found_tags = membership_candidates[k][g]
            found_tag_IDs = [t[1] for t in found_tags]
            if len(tID_list) != len(found_tags) or set(found_tag_IDs) != set(tID_list):
                keys_to_drop.append([k, g])

    for kg in keys_to_drop:
        if verbose:
            print("dropping", kg, membership_candidates[kg[0]][kg[1]])
        membership_candidates[kg[0]].pop(kg[1], None)

    return membership_candidates


def print_legal_multitags(legal_multitags: dict) -> None:
    """
    Prints the list of detected legal multitags and their groupings.

    Args:
        legal_multitags (dict): Dictionary of legal multitags with their groupings.
    """
    print(" M  G   [detection, ID] pairs")
    print("-- --   ---------------------------------")
    for mt in legal_multitags:
        if 0:
            print(legal_multitags[mt])
        else:
            for g in legal_multitags[mt]:
                print(f"{mt:2} {g:2}  ", legal_multitags[mt][g])
