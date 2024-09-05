import numpy as np
import os
import cv2

def save_camera(
        mtx, dist, rvecs, tvecs,
        img_h, img_w,
        newcameramtx, roi,
        calibdir='.',
        filename='camera.npz'
    ): 

    """
    Saves camera calibration parameters to a file.

    Args:
        mtx (np.ndarray): Camera matrix.
        dist (np.ndarray): Distortion coefficients.
        rvecs (np.ndarray): Rotation vectors.
        tvecs (np.ndarray): Translation vectors.
        img_h (int): Image height.
        img_w (int): Image width.
        newcameramtx (np.ndarray): New camera matrix.
        roi (list or np.ndarray): Region of interest (x, y, w, h).
        calibdir (str): Directory to save the file.
        filename (str): Name of the file.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """

    filepath = os.path.join(calibdir, filename)
    if os.path.isfile(filepath):
        print(f'Warning: File {filename} already exists. Not overwriting.')
        return False
    
    np.savez(filepath,
             mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs,
             img_h=img_h, img_w=img_w,
             newcameramtx=newcameramtx, roi=roi
             )   
    return True

def load_camera(calibdir, filename='camera.npz', ret='all'):
    """
    Loads camera calibration parameters from a file.

    Args:
        calibdir (str): Directory to load the file from.
        filename (str): Name of the file.
        ret (str): Specifies what to return ('all', 'each', or 'some').

    Returns:
        dict or tuple: Loaded camera parameters depending on `ret` value.
    """
    filepath = os.path.join(calibdir, filename)
    if not os.path.isfile(filepath):
        print(f'Warning: File {filename} does not exist.')
        return False

    npzfile = np.load(filepath)

    if ret == 'all':
        return npzfile
    elif ret == 'each':
        return (npzfile['mtx'], npzfile['dist'], npzfile['rvecs'], npzfile['tvecs'],
                npzfile['img_h'], npzfile['img_w'], npzfile['newcameramtx'], npzfile['roi'])
    elif ret == 'some':
        return npzfile['mtx'], npzfile['dist']

    return None

def undistort(img, mtx, dist, newcameramtx, roi=None):
    """
    Undistorts an image using camera calibration parameters.

    Args:
        img (np.ndarray): Input image.
        mtx (np.ndarray): Camera matrix.
        dist (np.ndarray): Distortion coefficients.
        newcameramtx (np.ndarray): New camera matrix.
        roi (list or np.ndarray, optional): Region of interest (x, y, w, h). Defaults to None.

    Returns:
        np.ndarray: Undistorted image.
    """
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    if roi:
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
    return dst

def camera_matrix_from_params(params):
    """
    Constructs a camera matrix from individual parameters.

    Args:
        params (list or np.ndarray): List or array of parameters [fx, fy, x0, y0].

    Returns:
        np.ndarray: 3x3 camera matrix.
    """
    return np.array([
        [params[0], 0.00, params[2]],
        [0.00, params[1], params[3]],
        [0.00, 0.00, 1.00]
    ])

def camera_params_from_matrix(matrix):
    """
    Extracts camera parameters from a camera matrix.

    Args:
        matrix (np.ndarray): 3x3 camera matrix.

    Returns:
        list: List of parameters [fx, fy, x0, y0].
    """
    return [matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]]
