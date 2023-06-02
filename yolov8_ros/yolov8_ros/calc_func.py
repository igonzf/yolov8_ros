#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import struct
import pandas as pd

def point2dto3d(pcl, x, y):
    """
    Converts the 2D coordinates of a point in a point cloud to 3D coordinates.

    Args:
        pcl (PointCloud): A PointCloud object containing the point cloud.
        x (int): The x-coordinate of the point in the point cloud.
        y (int): The y-coordinate of the point in the point cloud.

    Returns:
        list: A list of 3D coordinates [x, y, z] of the point.
    """

    is_bigendian = pcl.is_bigendian
    point_size = pcl.point_step
    row_step = pcl.row_step
            
    data_format = ""
            
    if is_bigendian:
        data_format = ">f"
    else:
        data_format = "<f"

    if x >= 0 and y >= 0 and (y * row_step) + (x * point_size) + point_size <= len(pcl.data):
        xp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size))[0]
        yp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size) + 4)[0]
        zp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size) + 8)[0]
        return [xp, yp, zp]
    else:
        return [0.0, 0.0, 0.0]

def get_intrinsic_matrix(camera_info):
    """
    Retrieves the intrinsic matrix from the camera information.

    Args:
        camera_info (CameraInfo): The camera information object.

    Returns:
        numpy.ndarray: The intrinsic matrix.
    """

    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    intrinsic_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
    
    return intrinsic_matrix
