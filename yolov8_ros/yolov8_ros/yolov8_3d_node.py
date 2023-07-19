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

import math
from array import *
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node

from sensor_msgs_py import point_cloud2
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, BoundingBox2D, BoundingBox3D
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
from yolov8_msgs.msg import Keypoint2D, Keypoint3D, PersonKeypoints2D, PersonKeypoints3D, PersonKeypoints2DArray, PersonKeypoints3DArray
from geometry_msgs.msg import Pose

from message_filters import Subscriber, ApproximateTimeSynchronizer

from .calc_func import point2dto3d
from .rviz_makers import create_marker_point, create_marker_cube

class Yolo3DNode(Node):

    def __init__(self) -> None:
        super().__init__("yolov8_3d_node")

        # topics
        self._markers_pub = self.create_publisher(MarkerArray, "markers", 10)
        self._keypoints3D_pub = self.create_publisher(PersonKeypoints3DArray, "keypoints3D", 10)
        self._detections3D_pub = self.create_publisher(Detection3DArray, "detections3D", 10)

        self._kpts_sub = Subscriber(
            self, PersonKeypoints2DArray, "keypoints"
        )
        self._bbox_sub = Subscriber(
            self, Detection2DArray, "detections"
        )
        self._person_bbox_sub = Subscriber(
            self, Detection2DArray, "person_detections"
        )
        self._pcl_sub = Subscriber(
            self, PointCloud2, "depth_points", qos_profile= qos_profile_sensor_data
        )
        
        tss_k = ApproximateTimeSynchronizer([self._kpts_sub, self._pcl_sub], 30, 0.1)
        tss_k.registerCallback(self.keypoints_cb)
        tss_bbox = ApproximateTimeSynchronizer([self._bbox_sub, self._pcl_sub], 30, 0.1)
        tss_bbox.registerCallback(self.bbox_cb)
        tss_person_bbox = ApproximateTimeSynchronizer([self._person_bbox_sub, self._pcl_sub], 30, 0.1)
        tss_person_bbox.registerCallback(self.bbox_cb)
    
    def get_3d_bbox(self, pcl: PointCloud2, bbox_2d: BoundingBox2D) -> BoundingBox3D:
        
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

        self.get_logger().info(f'{bbox_3d.size.x}, {bbox_3d.size.y}, {bbox_3d.size.z}')

        return bbox_3d

    def keypoints_cb(self, msg_kpts: PersonKeypoints2DArray, msg_pcl: PointCloud2) -> None:
         
        keypoints_3d = PersonKeypoints3DArray()
        markers = MarkerArray()

        for p_kpts in msg_kpts.keypoints:

            person_kpts = PersonKeypoints3D()

            for i, k in enumerate(p_kpts.keypoint_array):
                x_coord, y_coord = k.x, k.y

                kpt_3d = Keypoint3D()
                kpt_3d.x, kpt_3d.y, kpt_3d.z = point2dto3d(msg_pcl, x_coord, y_coord)

                person_kpts.keypoint_array.append(kpt_3d)
                
                marker = create_marker_point(msg_pcl.header.frame_id, kpt_3d.x, kpt_3d.y, kpt_3d.z)
                marker.header.stamp = msg_pcl.header.stamp
                marker.id = len(markers.markers)
                markers.markers.append(marker)

            keypoints_3d.keypoints.append(person_kpts)

        
        keypoints_3d.header.stamp = msg_kpts.header.stamp
        keypoints_3d.header.frame_id = msg_kpts.header.frame_id
        self._keypoints3D_pub.publish(keypoints_3d)
        self._markers_pub.publish(markers)
    
    def bbox_cb(self, msg_bbox: Detection2DArray, msg_pcl: PointCloud2) -> None:

        detections_3d = Detection3DArray()
        markers = MarkerArray()

        for detection in msg_bbox.detections:

            detection_3d = Detection3D()
            detection_3d.header = detection.header
            detection_3d.results = detection.results
            detection_3d.id = detection.id
            detection_3d.bbox = self.get_3d_bbox(msg_pcl, detection.bbox)
            marker = create_marker_cube(msg_pcl.header.frame_id, detection_3d.bbox)
            marker.header.stamp = msg_pcl.header.stamp
            marker.id = len(markers.markers)
            markers.markers.append(marker)

            detections_3d.detections.append(detection_3d)

        self._detections3D_pub.publish(detections_3d)
        self._markers_pub.publish(markers)

def main():
    rclpy.init()
    node = Yolo3DNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
