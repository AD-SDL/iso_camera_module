"""
pose_utils.py

Utilities for working with 3D pose and plane geometry, including:
- Conversion between rotation vectors and 4x4 pose matrices.
- Calculation of plane normals and distances from points to planes.
- Statistics for triangles formed by points in 3D space.
- Extraction of coordinate axes from pose and rotation vectors.

Functions:
- rvec_tvec_from_pose: Extract rotation and translation vectors from a 4x4 pose matrix.
- pose_from_rvec_tvec: Construct a 4x4 pose matrix from rotation and translation vectors.
- plane_norm: Compute the normal vector of a plane defined by three points.
- point_to_plane_norm: Compute the signed distance from a point to a plane.
- point_to_plane: Compute the signed distance from a point to a plane defined by three points.
- points_plane_stats: Compute and print statistics for all triangles formed by points.
- xyz_axes_from_pose: Extract the x, y, and z axes from a 4x4 pose matrix.
- xyz_axes_from_rvec: Extract the x, y, and z axes from a rotation vector.
- point_to_plane_rvec: Compute the signed distance from a point to a plane defined by a translation vector and a rotation vector.
"""

import cv2
import numpy as np


def rvec_tvec_from_pose(pose):
    """
    Extract rotation vector and translation vector from a 4x4 pose matrix.

    Args:
        pose (numpy.ndarray): A 4x4 pose matrix.

    Returns:
        tuple: Rotation vector and translation vector.
    """
    Rvec, _ = cv2.Rodrigues(pose[:3, :3])
    Rvec = Rvec.T[0]
    Tvec = pose[:3, 3]
    return Rvec, Tvec


def pose_from_rvec_tvec(rvec, tvec):
    """
    Construct a 4x4 pose matrix from a rotation vector and translation vector.

    Args:
        rvec (numpy.ndarray): Rotation vector.
        tvec (numpy.ndarray): Translation vector.

    Returns:
        numpy.ndarray: A 4x4 pose matrix.
    """
    Pose = np.zeros([4, 4])
    Pose[:3, :3], _ = cv2.Rodrigues(rvec)
    Pose[:3, 3] = tvec
    Pose[3, 3] = 1  # Ensure the matrix is homogeneous
    return Pose


def plane_norm(p0, p1, p2):
    """
    Compute the normal vector of a plane defined by three points.

    Args:
        p0, p1, p2 (numpy.ndarray): Points defining the plane.

    Returns:
        numpy.ndarray: Normal vector of the plane.
    """
    n = np.cross(p1 - np.array(p0), p2 - np.array(p0))
    return n / np.linalg.norm(n)


def point_to_plane_norm(p0, n, p):
    """
    Compute the signed distance from a point to a plane.

    Args:
        p0 (numpy.ndarray): A point on the plane.
        n (numpy.ndarray): Normal vector of the plane.
        p (numpy.ndarray): Point from which the distance is calculated.

    Returns:
        float: Signed distance from the point to the plane.
    """
    dist = np.dot(p - np.array(p0), n)
    return dist


def point_to_plane(p0, p1, p2, p):
    """
    Compute the signed distance from a point to a plane defined by three points.

    Args:
        p0, p1, p2 (numpy.ndarray): Points defining the plane.
        p (numpy.ndarray): Point from which the distance is calculated.

    Returns:
        float: Signed distance from the point to the plane.
    """
    n = plane_norm(p0, p1, p2)
    dist = point_to_plane_norm(p0, n, p)
    return dist


def points_plane_stats(pts):
    """
    Compute statistics for all triangles formed by points, including plane alignment, area, and distances.

    Args:
        pts (list of numpy.ndarray): List of points.

    Prints:
        Details of each triangle including plane normal, alignment, area, edge lengths, and point-to-plane distances.
    """
    first_time = True
    for i0, p0 in enumerate(pts):
        for i1, p1 in enumerate(pts):
            if i1 <= i0:
                continue
            for i2, p2 in enumerate(pts):
                if i2 <= i1:
                    continue
                n = plane_norm(p0, p1, p2)
                l0 = np.linalg.norm(p1 - np.array(p0))
                l1 = np.linalg.norm(p2 - np.array(p1))
                l2 = np.linalg.norm(p0 - np.array(p2))
                area = np.linalg.norm(np.cross(p1 - np.array(p0), p2 - np.array(p0))) / 2
                if first_time:
                    n0 = n
                    first_time = False
                alignment = np.sign(np.dot(n, n0))
                dotstr = f"{np.dot(n, n0):6.2f}"
                n = n * alignment
                z = np.array([np.dot(p - np.array(p0), n) for p in pts])
                zrms = np.linalg.norm(z) / 3
                zstr = "[" + "".join([f"{100*v:6.2f}" for v in z]) + "]"
                nstr = "[" + "".join([f"{v:6.2f}" for v in n]) + "]"
                print(
                    f"{i0:2} {i1:2} {i2:2} {alignment:3.0f} {100*area:6.2f} {l0:6.2f} {l1:6.2f} {l2:6.2f}",
                    nstr,
                    dotstr,
                    zstr,
                    f"{100*zrms:6.2f}",
                )
    return


def xyz_axes_from_pose(pose):
    """
    Extract the x, y, and z axes from a 4x4 pose matrix.

    Args:
        pose (numpy.ndarray): A 4x4 pose matrix.

    Returns:
        numpy.ndarray: Array of x, y, and z axes.
    """
    return pose[:3, :3].T


def xyz_axes_from_rvec(rvec):
    """
    Extract the x, y, and z axes from a rotation vector.

    Args:
        rvec (numpy.ndarray): Rotation vector.

    Returns:
        numpy.ndarray: Array of x, y, and z axes.
    """
    pose, _ = cv2.Rodrigues(rvec)
    return pose.T


def point_to_plane_rvec(tvec, rvec, p):
    """
    Compute the signed distance from a point to a plane defined by the translation vector and rotation vector.

    Args:
        tvec (numpy.ndarray): Translation vector.
        rvec (numpy.ndarray): Rotation vector.
        p (numpy.ndarray): Point from which the distance is calculated.

    Returns:
        float: Signed distance from the point to the plane.
    """
    n = xyz_axes_from_rvec(rvec)[2]
    return point_to_plane_norm(tvec, n, p)
