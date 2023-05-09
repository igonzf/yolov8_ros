
import cv2
import torch
import random

import numpy as np
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
        self.get_logger().info('entra')
        points = point_cloud2.read_points_list(msg_pcl)
        
        for k_pose in msg_kpts.keypoints:
            for i, k in enumerate(k_pose.keypoint_array):
                x_coord, y_coord = k.x, k.y
                self.get_logger().info(f'{x_coord}, {y_coord}')
                center_point = points[int((y_coord * msg_pcl.width) + x_coord)]
                self.get_logger().info(f'{center_point}')
                """ marker = self.create_marker(msg_image.header.frame_id, center_point[0], center_point[1], center_point[2])
                marker.header.stamp = msg_image.header.stamp
                marker.id = len(markers.markers)
                markers.markers.append(marker) """


    def create_marker(self, frame, x, y, z) -> Marker:

        marker = Marker()
        marker.header.frame_id = frame
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = False

        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = float(z)

        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15

        marker.color.b = 1.0
        marker.color.a = 1.0

        marker.lifetime = Duration(seconds=1.0).to_msg()
        #marker.text = percept.class_name

        return marker
    
    def kpts(self, kpts, cv_image, msg_image, radius=5, kpt_line=True):
        """Plot keypoints on the image.
        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                    for human pose. Default is True.
        Note: `kpt_line=True` currently only supports human pose plotting.
        """
        #markers = MarkerArray()
        keypointArray = KeypointArray()
        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):

            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]
            keypoint = Keypoint()
            keypoint.x = int(x_coord)
            keypoint.y = int(y_coord)
            keypoint.confidence = float(k[2])
            keypointArray.keypoint_array.append(keypoint)
            if x_coord % cv_image.shape[1] != 0 and y_coord % cv_image.shape[0] != 0:
                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue
                #self.get_logger().info(f'{x_coord}, {y_coord}')
                #if self.pointcloud != None:
                    #center_point = self.points[int((y_coord * self.pointcloud.width) + x_coord)]
                    #self.get_logger().info(f'{center_point}')
                    """ marker = self.create_marker(msg_image.header.frame_id, center_point[0], center_point[1], center_point[2])
                    marker.header.stamp = msg_image.header.stamp
                    marker.id = len(markers.markers)
                    markers.markers.append(marker) """
         
                cv2.circle(cv_image, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)
        self.keypoints.keypoints.append(keypointArray)
        #self._markers_pub.publish(markers)
        

        if kpt_line:
            ndim = kpts.shape[-1]
            for i, sk in enumerate(self.skeleton):
                pos1 = (int(kpts[(sk[0] - 1), 0]), int(kpts[(sk[0] - 1), 1]))
                pos2 = (int(kpts[(sk[1] - 1), 0]), int(kpts[(sk[1] - 1), 1]))
                if ndim == 3:
                    conf1 = kpts[(sk[0] - 1), 2]
                    conf2 = kpts[(sk[1] - 1), 2]
                    if conf1 < 0.5 or conf2 < 0.5:
                        continue
                cv2.line(cv_image, pos1, pos2, [int(x) for x in self.limb_color[i]], thickness=2, lineType=cv2.LINE_AA)

        return cv_image

    def image_cb(self, msg: Image) -> None:

        if self.enable:

            cv_image = self.cv_bridge.imgmsg_to_cv2(msg)
            results = self.yolo.predict(
                source=cv_image,
                verbose=False,
                stream=False,
                conf=0.1,
            )


def main():
    rclpy.init()
    node = PointCloudPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
