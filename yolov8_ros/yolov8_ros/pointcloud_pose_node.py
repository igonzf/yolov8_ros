
import cv2
import torch
import random

import numpy as np
import struct
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from rclpy.duration import Duration

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, colors
from ultralytics.tracker import BOTSORT, BYTETracker
from ultralytics.tracker.trackers.basetrack import BaseTrack
from ultralytics.yolo.utils import IterableSimpleNamespace, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_yaml
from sensor_msgs_py import point_cloud2

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import SetBool
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from yolov8_msgs.msg import Keypoint, KeypointArray, Keypoints

from message_filters import Subscriber, ApproximateTimeSynchronizer

class PointCloudPoseNode(Node):

    def __init__(self) -> None:
        super().__init__("pointcloud_pose_node")

        

        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],[8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

        self._class_to_color = {}
        self.cv_bridge = CvBridge()


        # topcis
        self._markers_pub = self.create_publisher(MarkerArray, "markers", 10)

        self._kpts_sub = Subscriber(
            self, Keypoints, "keypoints"
        )
        self._pcl_sub = Subscriber(
            self, PointCloud2, "/camera/depth_registered/points", qos_profile= qos_profile_sensor_data
        )
        tss = ApproximateTimeSynchronizer([self._kpts_sub, self._pcl_sub], 30, 0.1)
        tss.registerCallback(self.pointcloud_pose_cb)

    def pointcloud_pose_cb(self, msg_kpts: Keypoints, msg_pcl: PointCloud2) -> None:

        markers = MarkerArray()
        is_bigendian = msg_pcl.is_bigendian
        point_size = msg_pcl.point_step
        row_step = msg_pcl.row_step
                
        data_format = ""
                
        if is_bigendian:
            data_format = ">f"
        else:
            data_format = "<f"

        for k_pose in msg_kpts.keypoints:
            for i, k in enumerate(k_pose.keypoint_array):
                x_coord, y_coord = k.x, k.y
                self.get_logger().info(f'{x_coord}, {y_coord}')
                #point = point_cloud2.read_points(msg_pcl, field_names=('x', 'y', 'z'), skip_nans=True, uvs=[x_coord, y_coord])[0]
                #center_point = point_cloud2.read_points(msg_pcl, field_names='z', skip_nans=False, uvs=[x_coord, y_coord])
                #self.get_logger().info(f'{point}')
                

                    
                xp = struct.unpack_from(data_format, msg_pcl.data, (y_coord * row_step) + (x_coord * point_size))[0]
                yp = struct.unpack_from(data_format, msg_pcl.data, (y_coord * row_step) + (x_coord * point_size) + 4)[0]
                zp = struct.unpack_from(data_format, msg_pcl.data, (y_coord * row_step) + (x_coord * point_size) + 8)[0]


                marker = self.create_marker(msg_pcl.header.frame_id, xp, yp, zp)
                marker.header.stamp = msg_pcl.header.stamp
                marker.id = len(markers.markers)
                markers.markers.append(marker)
                
        self._markers_pub.publish(markers)

    def create_marker(self, frame, x, y, z) -> Marker:

        marker = Marker()
        marker.header.frame_id = frame
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.05
        marker.scale.y = 0.05
        marker.scale.z = 0.05

        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.lifetime = Duration(seconds=1.0).to_msg()
        #marker.text = percept.class_name

        return marker


def main():
    rclpy.init()
    node = PointCloudPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
