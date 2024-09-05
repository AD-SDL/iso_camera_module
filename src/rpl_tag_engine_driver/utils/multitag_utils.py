import cv2
import numpy as np

import camera_utils
import pose_utils
import apriltag_utils

def coplanar_test_one(rvec0, tvec0, rvec1, tvec1):
    """
    Test coplanarity between two poses.

    Args:
        rvec0, rvec1 (numpy.ndarray): Rotation vectors for the two poses.
        tvec0, tvec1 (numpy.ndarray): Translation vectors for the two poses.

    Returns:
        str: Formatted string with separation distance and coplanarity distances.
    """
    sepn = np.linalg.norm(tvec0 - tvec1)
    dist1 = pose_utils.point_to_plane_rvec(tvec0, rvec0, tvec1) / sepn
    dist0 = pose_utils.point_to_plane_rvec(tvec1, rvec1, tvec0) / sepn
    return f'{sepn:7.2f} {dist1:7.2f} {dist0:7.2f} ({np.linalg.norm([dist1,dist0]):7.2f})'

def coplanar_detection_test(detections, d_dex0, d_dex1):
    """
    Test coplanarity between two detections.

    Args:
        detections (list): List of detections.
        d_dex0, d_dex1 (int): Indices of the two detections to compare.

    Prints:
        Results of the coplanarity test between the two detections.
    """
    rvec0, tvec0 = pose_utils.rvec_tvec_from_pose(detections[d_dex0 * 4 + 1])
    print('     ', apriltag_utils.build_rstr(rvec0), apriltag_utils.build_tstr(tvec0))
    rvec1, tvec1 = pose_utils.rvec_tvec_from_pose(detections[d_dex1 * 4 + 1])
    print('     ',apriltag_utils.build_rstr(rvec1), apriltag_utils.build_tstr(tvec1))
    costr = coplanar_test_one(rvec0, tvec0, rvec1, tvec1)
    print(f'{d_dex0:2} {d_dex1:2}:', costr)
    return

def triangle_vs_tag_normal_test(detections, d_dex0, d_dex1, d_dex2):
    """
    Compare the normal of a triangle formed by tag origins with tag normals.

    Args:
        detections (list): List of detections.
        d_dex0, d_dex1, d_dex2 (int): Indices of the three detections forming the triangle.

    Prints:
        Results of the comparison between the triangle normal and the tag normals.
    """
    tvecs = [detections[d * 4 + 1][:3, 3] for d in [d_dex0, d_dex1, d_dex2]]
    triangle_normal = pose_utils.plane_norm(tvecs[0], tvecs[1], tvecs[2])
    tag_normals = [pose_utils.xyz_axes_from_pose(detections[d * 4 + 1][:3, :3])[2] for d in [d_dex0, d_dex1, d_dex2]]
    print(apriltag_utils.build_rstr(triangle_normal))
    for tn in tag_normals:
        print(apriltag_utils.build_rstr(tn), f'{np.dot(tn, triangle_normal):7.2f}')
    return

def triangle_tag_normal_report(rvecs, tvecs, verbose=False):
    """
    Report the comparison between a triangle normal and tag normals.

    Args:
        rvecs (list of numpy.ndarray): List of rotation vectors for the tags.
        tvecs (list of numpy.ndarray): List of translation vectors for the tags.
        verbose (bool): If True, include detailed information.

    Prints:
        Comparison between the triangle normal and each tag normal.
    """
    triangle_normal = pose_utils.plane_norm(tvecs[0].flatten(), tvecs[1].flatten(), tvecs[2].flatten())
    tag_normals = [pose_utils.xyz_axes_from_rvec(rvec.flatten())[2] for rvec in rvecs]
    print(apriltag_utils.build_rstr(triangle_normal))
    for i, tn in enumerate(tag_normals):
        if verbose:
            rv = rvecs[i].flatten()
            print(apriltag_utils.build_rstr(rv), apriltag_utils.build_rstr(tn),
                  f'{np.dot(tn, triangle_normal):7.2f}', f'{np.linalg.norm(rv) * 180 / np.pi:8.2f}')
        else:
            print(apriltag_utils.build_rstr(tn), f'{np.dot(tn, triangle_normal):7.2f}')
    return

def polygon_orientation(pts, image_sense=False):
    """
    Determine if the vertices of a polygon are ordered clockwise or counterclockwise.

    Args:
        pts (list of numpy.ndarray): List of vertices of the polygon.
        image_sense (bool): If True, interpret the order as clockwise in an image (y increasing top to bottom).

    Returns:
        bool: True if vertices are ordered predominantly clockwise, False otherwise.
    """
    sign = -1 if image_sense else 1
    N = len(pts)
    clockwise = 0
    for i in range(N):
        x1 = pts[i]
        x2 = pts[(i + 1) % N]
        clockwise += sign * (x2[0] - x1[0]) * (x2[1] + x1[1])
    return clockwise >= 0

def detections_subset(detections, d_dex_list):
    """
    Gather a subset of detections based on provided indices.

    Args:
        detections (list): List of detections.
        d_dex_list (list of int): List of indices of detections to include.

    Returns:
        list: Subset of detections including all associated data.
    """
    det_subset = []
    for d in d_dex_list:
        for doff in range(4):
            det_subset.append(detections[d + doff])
    return det_subset

def corners_subset(detections, d_dex_list):
    """
    Gather the corner points from a subset of detections.

    Args:
        detections (list): List of detections.
        d_dex_list (list of int): List of indices of detections to include.

    Returns:
        list: List of corner points for each detection.
    """
    tag_corners = []
    for d in d_dex_list:
        tag_corners.append(apriltag_utils.value_by_field(detections[4 * d], 'Corners'))
    return tag_corners

def field_subset(detections, d_dex_list, field):
    """
    Gather specific field values from a subset of detections.

    Args:
        detections (list): List of detections.
        d_dex_list (list of int): List of indices of detections to include.
        field (str): Name of the field to extract from each detection.

    Returns:
        list: List of values for the specified field from each detection.
    """
    tag_fields = []
    for d in d_dex_list:
        tag_fields.append(apriltag_utils.value_by_field(detections[4 * d], field))
    return tag_fields

def pose_rotation_ambiguity_test(recon_errors, contrast_threshold=0.2):
    """
    Test for rotation ambiguity in pose estimation.

    Args:
        recon_errors (numpy.ndarray): Array of reconstruction errors from IPPE.
        contrast_threshold (float): Threshold to determine if the contrast is significant enough.

    Returns:
        bool: True if the contrast between errors is above the threshold, False otherwise.
    """
    e0, e1 = recon_errors.flatten()
    contrast = np.abs(e0 - e1) / (e0 + e1)
    keep = contrast > contrast_threshold
    return keep

def consistent_coplanar_poses(tag_corners, tag_size, camera_params, distCoeffs=None, verbose=False, kit_and_kaboodle=False):
    """
    Find consistent and coplanar poses from tag detections.

    Args:
        tag_corners (list of numpy.ndarray): 2D coordinates of tag corners in the image.
        tag_size (float): Size of the tag in object coordinates.
        camera_params (tuple): Camera parameters (fx, fy, cx, cy).
        distCoeffs (numpy.ndarray, optional): Distortion coefficients. Defaults to None.
        verbose (bool, optional): If True, print additional debug information. Defaults to False.
        kit_and_kaboodle (bool, optional): If True, return all results for debugging. Defaults to False.

    Returns:
        tuple: Best rotation and translation vectors.
            - best_rvecs (list of numpy.ndarray): Best rotation vectors.
            - best_tvecs (list of numpy.ndarray): Best translation vectors.
            - find_best (dict, optional): Detailed results if `kit_and_kaboodle` is True.
            - results (list of lists, optional): All results if `kit_and_kaboodle` is True.
    """
    # 3D coordinates of tag corners
    opoints = apriltag_utils.tag_opoints(tag_size)
    cameraMatrix = camera_utils.camera_matrix_from_params(camera_params)

    results = []
    for these_corners in tag_corners:
        ret, rvecs, tvecs, errs = apriltag_utils.ippe(opoints, these_corners, cameraMatrix, distCoeffs=distCoeffs)
        results.append([ret, rvecs, tvecs, errs])

    # Compute plane normal from tag centers
    tvecs = [r[2][0].flatten() for r in results]
    triangle_normal = pose_utils.plane_norm(tvecs[0], tvecs[1], tvecs[2])
    centers = np.array([np.mean(poly, axis=0) for poly in tag_corners])
    
    if not polygon_orientation(centers, image_sense=True):
        triangle_normal = -triangle_normal

    if verbose:
        for tv in tvecs:
            print(tv)
        print('Normal to oriented plane of tag group:', triangle_normal)

    # Sanity check: ensure tags are correctly oriented
    for i, corns in enumerate(tag_corners):
        if not polygon_orientation(corns, image_sense=True):
            print(f'Tag {i} in set is not oriented correctly. This could indicate a mirrored image or a tragic error.')
            print('WARNING: Ignoring this tag and continuing with the results.')

    # Determine the best pose solution
    rvecs = [
        [r[1][0].flatten() for r in results],
        [r[1][1].flatten() for r in results]
    ]
    tag_normals = [
        [pose_utils.xyz_axes_from_rvec(rv)[2] for rv in rvecs[0]],
        [pose_utils.xyz_axes_from_rvec(rv)[2] for rv in rvecs[1]]
    ]
    dots = [
        [np.dot(t, triangle_normal) for t in tag_normals[0]],
        [np.dot(t, triangle_normal) for t in tag_normals[1]]
    ]
    tvecs = [
        [r[2][0].flatten() for r in results],
        [r[2][1].flatten() for r in results]
    ]

    best = [0 for _ in results]
    for i in range(len(results)):
        if dots[1][i] > dots[0][i]:
            best[i] = 1

    find_best = {
        'rvecs': rvecs,
        'tvecs': tvecs,
        'tag_normals': tag_normals,
        'dots': dots,
        'best_index': best
    }

    if verbose:
        print('-' * 60, 'Dot products using tag normal from IPPE rvecs')
        for t0, t1 in zip(tag_normals[0], tag_normals[1]):
            print(f'{np.dot(triangle_normal, t0):6.2f} {np.dot(triangle_normal, t1):6.2f}')
        print()
        apriltag_utils.triangle_tag_normal_report(rvecs, tvecs)
        print()
        axes = [pose_utils.xyz_axes_from_rvec(r[1][0].flatten()) for r in results]
        for a in axes:
            print(apriltag_utils.build_rstr(a[0]), apriltag_utils.build_rstr(a[1]), apriltag_utils.build_rstr(a[2]))

    best_rvecs = [rvecs[b][i] for i, b in enumerate(best)]
    best_tvecs = [tvecs[b][i] for i, b in enumerate(best)]

    if kit_and_kaboodle:
        return best_rvecs, best_tvecs, find_best, results

    return best_rvecs, best_tvecs

def axes_and_pose_box_coords(rvec, tvec, camera_params, tag_size, z_sign=1):
    """
    Compute the 2D coordinates of axes and pose box edges projected onto the image.

    Args:
        rvec (numpy.ndarray): Rotation vector of the tag.
        tvec (numpy.ndarray): Translation vector of the tag.
        camera_params (tuple): Camera parameters (fx, fy, cx, cy).
        tag_size (float): Size of the tag in object coordinates.
        z_sign (int, optional): Sign for the z-axis length. Defaults to 1.

    Returns:
        tuple: 
            - axes_coords (list of numpy.ndarray): Coordinates of the tag's axes in image space.
            - box_edge_coords (list of numpy.ndarray): Coordinates of the tag's pose box edges in image space.
    """
    # Camera matrix K from camera parameters
    fx, fy, cx, cy = camera_params
    cameraMatrix = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
    dcoeffs = np.zeros(5)

    # Pose box coordinates
    opoints = np.array([
        -1, -1, 0,
         1, -1, 0,
         1,  1, 0,
        -1,  1, 0,
        -1, -1, -2 * z_sign,
         1, -1, -2 * z_sign,
         1,  1, -2 * z_sign,
        -1,  1, -2 * z_sign,
    ]).reshape(-1, 1, 3) * 0.5 * tag_size

    edges = np.array([
        0, 1,        1, 2,        2, 3,        3, 0,
        0, 4,        1, 5,        2, 6,        3, 7,
        4, 5,        5, 6,        6, 7,        7, 4
    ]).reshape(-1, 2)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, cameraMatrix, dcoeffs)

    box_edge_coords = []
    for i, j in edges:
        coords = np.array([ipoints.squeeze()[i], ipoints.squeeze()[j]]).T
        box_edge_coords.append(coords)

    # Axes coordinates
    opoints = np.float32([
        [0, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ]) * tag_size

    edges = np.array([
        0, 1,
        0, 2,
        0, 3
    ]).reshape(-1, 2)

    ipoints, _ = cv2.projectPoints(opoints, rvec, tvec, cameraMatrix, dcoeffs)

    axes_coords = []
    for i, j in edges:
        coords = np.array([ipoints.squeeze()[i], ipoints.squeeze()[j]]).T
        axes_coords.append(coords)

    return axes_coords, box_edge_coords

def tag_opoints_offset(tag_size, offset=[0, 0]):
    """
    Compute the 3D coordinates of tag corners with an offset.

    Args:
        tag_size (float): Size of the tag in object coordinates.
        offset (list of float, optional): Offset for the tag corners. Defaults to [0, 0].

    Returns:
        numpy.ndarray: 3D coordinates of the tag corners with the given offset.
    """
    ox = offset[0] * 2      # Compensate for * 0.5
    oy = offset[1] * 2
    opoints = np.array([
        -1 + ox, -1 + oy, 0,
         1 + ox, -1 + oy, 0,
         1 + ox,  1 + oy, 0,
        -1 + ox,  1 + oy, 0
    ]).reshape(-1, 1, 3) * 0.5 * tag_size
    return opoints

def multitag_opoints(tag_size, offsets):
    """
    Compute the combined 3D coordinates of multiple tags with offsets.

    Args:
        tag_size (float): Size of each tag in object coordinates.
        offsets (list of lists): Offsets for each tag.

    Returns:
        numpy.ndarray: Combined 3D coordinates of all tags with their respective offsets.
    """
    opoints = tag_opoints_offset(tag_size, offsets[0])
    for o in offsets[1:]:
        opoints = np.vstack([opoints, tag_opoints_offset(tag_size, o)])
    return opoints

def many_tag_ippe(tag_corners, tag_size, cameraMatrix, distCoeffs=None):
    """
    Compute IPPE (Iterative Perspective-n-Point Estimation) for multiple tags.

    Args:
        tag_corners (list of numpy.ndarray): 2D image coordinates of tag corners.
        tag_size (float): Size of the tag in object coordinates.
        cameraMatrix (numpy.ndarray): Camera matrix.
        distCoeffs (numpy.ndarray, optional): Distortion coefficients. Defaults to None.

    Returns:
        list of lists: Results of IPPE for each set of tag corners.
    """
    opoints = apriltag_utils.tag_opoints(tag_size)
    results = []
    for these_corners in tag_corners:
        ret, rvecs, tvecs, errs = apriltag_utils.ippe(opoints, these_corners, cameraMatrix, distCoeffs=distCoeffs)
        results.append([ret, rvecs, tvecs, errs])
    return results

def value_from_all_tags(detections, field):
    """
    Extract values from all detected tags based on the specified field.

    Args:
        detections (list): List of detections including apriltag detections and pose information.
        field (str): Field name to extract values from.

    Returns:
        list: Extracted values for the specified field.
    """
    stride = 4
    if not detections:
        return []

    if len(detections[0]) != 8:
        return []

    if len(detections[1]) != 4:
        if field == 'Pose':
            return []
        stride = 1

    results = []
    if field == 'Pose':
        for p in detections[1::4]:
            results.append(p)
    else:
        for d in detections[0::4]:
            results.append(apriltag_utils.value_by_field(d, field))

    return results

def format_list(nlist):
    """
    Format a list of numbers into a string representation with space-separated values enclosed in square brackets.

    Args:
        nlist (list of float): List of numbers to format.

    Returns:
        str: Formatted string representation of the list.
    """
    return '[' + ' '.join([f'{v:0.0f}' for v in nlist]) + ']'

def find_tag_neighbors(these_corners, dthresh=0.1, sep_factor=2.0, verbose=False):
    """
    Find neighboring tags based on their size and separation distance.

    Args:
        these_corners (list of numpy.ndarray): List of 2D corner coordinates for each tag.
        dthresh (float, optional): Threshold for diameter difference to consider tags as neighbors. Defaults to 0.1.
        sep_factor (float, optional): Factor to multiply the average diameter to determine separation distance. Defaults to 2.0.
        verbose (bool, optional): Print debug information if True. Defaults to False.

    Returns:
        list of lists: Pairs of indices representing neighboring tags.
    """
    these_centers = [np.mean(c, axis=0) for c in these_corners]
    these_diams = [max(np.linalg.norm(c[0] - c[2]), np.linalg.norm(c[1] - c[3])) for c in these_corners]

    tag_neighbors = []
    for i1, (d1, cen1, corns1) in enumerate(zip(these_diams, these_centers, these_corners)):
        for i2, (d2, cen2, corns2) in enumerate(zip(these_diams, these_centers, these_corners)):
            if i2 <= i1:
                continue
            if np.abs(d1 - d2) / (d1 + d2) > dthresh:
                continue
            if np.linalg.norm(cen1 - cen2) < sep_factor * np.mean([d1, d2]):
                tag_neighbors.append([i1, i2])
                if verbose:
                    print(f'{i1:2} {i2:2} {d1:3.0f} {d2:3.0f}')
    return tag_neighbors

def breadth_first_connected_components(graph, nodes):
    """
    Find all connected components in a graph using breadth-first search.

    Args:
        graph (dict): Dictionary where each key is a node and the value is a list of connected nodes.
        nodes (list): List of nodes in the graph.

    Returns:
        list of sets: List of connected components, each represented as a set of nodes.
    """
    seen = set()
    result = []   # List of sets to hold the final result
    for node in nodes:
        if node not in seen:
            components = set()
            leaves = [node]
            while leaves:
                leaf = leaves.pop()
                seen.add(leaf)
                components.add(leaf)
                for connected_node in graph[leaf]:
                    if connected_node not in seen:
                        leaves.append(connected_node)
            result.append(components)
    return result

def aggregate_tag_clusters(edges):
    """
    Aggregate tags into clusters based on their connections.

    Args:
        edges (list of lists): List of edges where each edge is a pair of connected tags.

    Returns:
        list of sets: List of clusters of tags, each represented as a set of tag indices.
    """
    graph = dict()
    nodes = set()
    for e in edges:
        if e[0] not in graph:
            graph[e[0]] = [e[0]]
        if e[1] not in graph:
            graph[e[1]] = [e[1]]
        graph[e[1]].append(e[0])
        graph[e[0]].append(e[1])
        nodes.add(e[0])
        nodes.add(e[1])
    
    multitag_clusters = breadth_first_connected_components(graph, list(nodes))
    return multitag_clusters

def tag_grid_from_basis(p_minx, basis, tag_nums, pts):
    """
    Generate a grid of tag indices based on model points created using a basis and minimum point.

    Args:
        p_minx (numpy.ndarray): Minimum point in the grid.
        basis (list of numpy.ndarray): Two basis vectors defining the grid orientation.
        tag_nums (list of int): List of tag numbers corresponding to the points.
        pts (list of numpy.ndarray): List of actual points corresponding to the tags.

    Returns:
        list of lists: Each entry contains grid indices and the corresponding tag number.
    """
    minxy = np.min(pts, axis=0)
    maxxy = np.max(pts, axis=0)
    tag_grid = []
    tag_nums_to_assign = set(tag_nums)
    
    for i in range(3):
        for j in range(2):
            gen_point = p_minx + i * basis[0] + j * basis[1]
            best_so_far = np.linalg.norm(maxxy - minxy) + 1
            best_assignment = -1
            
            for k in range(len(pts)):
                tn = tag_nums[k]
                if tn not in tag_nums_to_assign:
                    continue
                this_dist = np.linalg.norm(pts[k] - gen_point)
                if this_dist < best_so_far:
                    best_so_far = this_dist
                    best_assignment = tn
            
            tag_grid.append([i, j, best_assignment])
    
    return tag_grid

def find_2x3_grid(tag_nums, pts, threshold=0.1, verbose=False):
    """
    Find a 2x3 grid of tags from a list of 6 points and their corresponding tag numbers.

    Args:
        tag_nums (list of int): List of tag numbers corresponding to the points.
        pts (list of numpy.ndarray): List of 6 points to form the grid.
        threshold (float, optional): Threshold for validating the basis vectors. Defaults to 0.1.
        verbose (bool, optional): Print debug information if True. Defaults to False.

    Returns:
        bool: True if a valid 2x3 grid was found, otherwise False.
        list of lists or None: The tag grid if successful, otherwise None.
        list of numpy.ndarray or None: The basis vectors if successful, otherwise None.
    """
    if len(pts) != 6:
        print('FAIL: need exactly six points')
        return False, None, None

    minxy = np.min(pts, axis=0)
    maxxy = np.max(pts, axis=0)

    minx_dex = np.where(pts[:, 0] == minxy[0])[0][0]
    p_minx = pts[minx_dex]
    maxx_dex = np.where(pts[:, 0] == maxxy[0])[0][0]
    p_maxx = pts[maxx_dex]
    
    info = []
    for i, p in enumerate(pts):
        if i == minx_dex:
            continue
        delta = p - pts[minx_dex]
        length = np.linalg.norm(delta)
        theta = np.arctan2(delta[0], delta[1])
        info.append([i, delta, length, theta])
        if verbose:
            print(f'{i:3} {delta[0]:6.1f} {delta[1]:6.1f} {length:6.1f} {theta * 180. / np.pi:6.1f}')
    
    three_closest = np.argsort([r[2] for r in info])[:3]

    predict_this_distance = max([r[2] for r in info])
    possible_bases = []
    
    for i in three_closest:
        for j in three_closest:
            if i == j:
                continue
            candidate_furthest_delta = 2 * info[i][1] + info[j][1]
            candidate_furthest_distance = np.linalg.norm(candidate_furthest_delta)
            furthest_test = np.linalg.norm(p_minx + candidate_furthest_delta - p_maxx) / candidate_furthest_distance
            contrast = abs(candidate_furthest_distance - predict_this_distance) / (candidate_furthest_distance + predict_this_distance)
            
            if furthest_test < threshold:
                basis = [info[i][1], info[j][1]]
                tag_grid = tag_grid_from_basis(p_minx, basis, tag_nums, pts)
                return True, tag_grid, basis
            
            possible_bases.append([i, j, candidate_furthest_distance, contrast])
            if verbose:
                print(f'{i:3} {j:3} {candidate_furthest_distance:6.1f} {contrast:5.2f} {furthest_test:5.3f}')
    
    return False, None, None

def fmt_this_my_way(tag_grid, basis):
    """
    Format the tag grid and basis vectors for display.

    Args:
        tag_grid (list of lists): The tag grid with indices and tag numbers.
        basis (list of numpy.ndarray): The basis vectors.

    Returns:
        str: Formatted string of the tag grid and basis vectors.
    """
    def p1(vec):
        return '[' + ' '.join([f'{v:2}' for v in vec]) + ']'
    
    def p2(vec):
        return '[' + ' '.join([f'{v:5.1f}' for v in vec]) + ']'
    
    s1 = '[' + ', '.join([p1(vec) for vec in tag_grid]) + ']'
    s2 = '[' + ', '.join([p2(vec) for vec in basis]) + ']'
    return s1 + '   ' + s2


