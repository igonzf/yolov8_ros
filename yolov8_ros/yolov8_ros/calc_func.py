#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2023  Irene González Fernández

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

''' Two functions:
def point_plane_distance
def point_3d_line_distance
def point2dto3d
def get_intrinsic_matrix
'''

import math
import numpy as np
import struct
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointCloud2
from vision_msgs.msg import BoundingBox2D, BoundingBox3D
from geometry_msgs.msg import Pose


def point2dto3d(pcl, x, y):

    is_bigendian = pcl.is_bigendian
    point_size = pcl.point_step
    row_step = pcl.row_step
            
    data_format = ""
            
    if is_bigendian:
        data_format = ">f"
    else:
        data_format = "<f"

    xp = yp = zp = 0.0

    if x >= 0 and y >= 0 and (y * row_step) + (x * point_size) + point_size <= len(pcl.data):
        xp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size))[0]
        yp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size) + 4)[0]
        zp = struct.unpack_from(data_format, pcl.data, (y * row_step) + (x * point_size) + 8)[0]

    return [xp, yp, zp]

def get_3d_bbox(pcl: PointCloud2, bbox_2d: BoundingBox2D) -> BoundingBox3D:
        
        x = int(bbox_2d.center.position.x)
        y = int(bbox_2d.center.position.y)

        x_min = y_min = z_min = float('inf')
        x_max = y_max = z_max = float('-inf')

        #center point bbox
        cp_x, cp_y, cp_z = point2dto3d(pcl, x, y)

        #bbox limits
        left_max_x, left_max_y, left_max_z = point2dto3d(pcl, int(bbox_2d.center.position.x - bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y + bbox_2d.size_y / 2.0))
        left_min_x, left_min_y, left_min_z = point2dto3d(pcl, int(bbox_2d.center.position.x - bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y - bbox_2d.size_y / 2.0))
        right_max_x, right_max_y, right_max_z = point2dto3d(pcl, int(bbox_2d.center.position.x + bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y + bbox_2d.size_y / 2.0))
        right_min_x, right_min_y, right_min_z = point2dto3d(pcl, int(bbox_2d.center.position.x + bbox_2d.size_x / 2.0),
                        int(bbox_2d.center.position.y - bbox_2d.size_y / 2.0))
        
        weight = (right_max_x - left_max_x) if not math.isnan((right_max_x - left_max_x)) else (right_min_x - left_min_x)
        high = (left_max_y - left_min_y) if not math.isnan((left_max_y - left_min_y)) else (right_max_y - right_min_y)

        for point in point_cloud2.read_points(pcl, field_names=("x", "y", "z"), skip_nans=True):
            dist_x = abs(point[0] - cp_x)
            dist_y = abs(point[1] - cp_y)

            #get the depth of the points near the center point
            if dist_x <=0.02 and dist_y <=0.02:
                z = float(point[2])
                z_min = min(z_min, z)
                z_max = max(z_max, z) 
        

        bbox_3d = BoundingBox3D()

        center_point = Pose()
        center_point.position.x = cp_x
        center_point.position.y = cp_y
        center_point.position.z = cp_z
        bbox_3d.center = center_point
        bbox_3d.size.x = weight
        bbox_3d.size.y = high
        bbox_3d.size.z = z_max - z_min

        return bbox_3d

def is_xy_in_bbox(xy, bbox) -> bool:
        x, y = xy[0], xy[1]
        xmin = round(bbox.center.position.x - bbox.size_x / 2.0)
        ymin = round(bbox.center.position.y - bbox.size_y / 2.0)
        xmax = round(bbox.center.position.x + bbox.size_x / 2.0)
        ymax = round(bbox.center.position.y + bbox.size_y / 2.0)
        if x >= xmin and x <= xmax and y >= ymin and y <= ymax: 
            return True
        return False

def get_intrinsic_matrix(camera_info):

    fx = camera_info.k[0]
    fy = camera_info.k[4]
    cx = camera_info.k[2]
    cy = camera_info.k[5]

    intrinsic_matrix = np.array([[fx, 0, cx],
                                [0, fy, cy],
                                [0, 0, 1]])
    
    return intrinsic_matrix

def get_euclidean_distance(p1, p2):
    return math.sqrt((p2.x -p1.x)**2 + (p2.y - p1.y)**2 + (p2.z - p1.z)**2)