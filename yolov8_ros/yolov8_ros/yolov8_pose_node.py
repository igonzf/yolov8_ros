
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

class Yolov8Node(Node):

    def __init__(self) -> None:
        super().__init__("yolov8_node")

        # params
        self.declare_parameter("model", "yolov8m.pt")
        model = self.get_parameter(
            "model").get_parameter_value().string_value

        self.declare_parameter("tracker", "bytetrack.yaml")
        tracker = self.get_parameter(
            "tracker").get_parameter_value().string_value

        self.declare_parameter("device", "cuda:0")
        device = self.get_parameter(
            "device").get_parameter_value().string_value

        self.declare_parameter("threshold", 0.5)
        self.threshold = self.get_parameter(
            "threshold").get_parameter_value().double_value

        self.declare_parameter("enable", True)
        self.enable = self.get_parameter(
            "enable").get_parameter_value().bool_value

        self.skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],[8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        self.kpt_color = colors.pose_palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        self.limb_color = colors.pose_palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]

        self._class_to_color = {}
        self.cv_bridge = CvBridge()
        self.tracker = self.create_tracker(tracker)
        self.yolo = YOLO(model)
        self.yolo.fuse()
        self.yolo.to(device)
        self.keypoints = Keypoints()


        # topcis
        self._pub = self.create_publisher(Detection2DArray, "detections", 10)
        self._dbg_pub = self.create_publisher(Image, "dbg_image", 10)
        self._kpts_pub = self.create_publisher(Keypoints, "keypoints", 10)
        self._sub = self.create_subscription(
            Image, "image_raw", self.image_cb,
            qos_profile_sensor_data
        )

        # services
        self._srv = self.create_service(SetBool, "enable", self.enable_cb)

    def create_tracker(self, tracker_yaml) -> BaseTrack:

        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in ["bytetrack", "botsort"], \
            f"Only support 'bytetrack' and 'botsort' for now, but got '{cfg.tracker_type}'"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=1)
        return tracker

    def enable_cb(self,
                  req: SetBool.Request,
                  res: SetBool.Response
                  ) -> SetBool.Response:
        self.enable = req.data
        res.success = True
        return res
    
    def kpts(self, kpts, cv_image, radius=5, kpt_line=True):
        """Plot keypoints on the image.
        Args:
            kpts (tensor): Predicted keypoints with shape [17, 3]. Each keypoint has (x, y, confidence).
            shape (tuple): Image shape as a tuple (h, w), where h is the height and w is the width.
            radius (int, optional): Radius of the drawn keypoints. Default is 5.
            kpt_line (bool, optional): If True, the function will draw lines connecting keypoints
                                    for human pose. Default is True.
        Note: `kpt_line=True` currently only supports human pose plotting.
        """
        keypointArray = KeypointArray()

        nkpt, ndim = kpts.shape
        is_pose = nkpt == 17 and ndim == 3
        kpt_line &= is_pose  # `kpt_line=True` for now only supports human pose plotting
        for i, k in enumerate(kpts):

            color_k = [int(x) for x in self.kpt_color[i]] if is_pose else colors(i)
            x_coord, y_coord = k[0], k[1]

            if x_coord % cv_image.shape[1] != 0 and y_coord % cv_image.shape[0] != 0:

                if len(k) == 3:
                    conf = k[2]
                    if conf < 0.5:
                        continue

                keypoint = Keypoint()
                keypoint.x = int(x_coord)
                keypoint.y = int(y_coord)
                keypoint.confidence = float(k[2])
                keypointArray.keypoint_array.append(keypoint)
                cv2.circle(cv_image, (int(x_coord), int(y_coord)), radius, color_k, -1, lineType=cv2.LINE_AA)

        self.keypoints.keypoints.append(keypointArray)
        

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

            if 'keypoints' in results[0].keys:
                self.keypoints = Keypoints()
                keypoints = results[0].keypoints
                self.keypoints.header.stamp = msg.header.stamp
                self.keypoints.header.frame_id = msg.header.frame_id
                
                for k in keypoints:
                    cv_image = self.kpts(k, cv_image)
                
                self._kpts_pub.publish(self.keypoints)
                
            # track
            det = results[0].boxes.cpu().numpy()

            if len(det) > 0:
                im0s = self.yolo.predictor.batch[2]
                im0s = im0s if isinstance(im0s, list) else [im0s]

                tracks = self.tracker.update(det, im0s[0])
                if len(tracks) > 0:
                    results[0].update(boxes=torch.as_tensor(tracks[:, :-1]))

            # create detections msg
            detections_msg = Detection2DArray()
            detections_msg.header = msg.header

            results = results[0].cpu()

            for b in results.boxes:

                label = self.yolo.names[int(b.cls)]
                score = float(b.conf)

                if score < self.threshold:
                    continue

                detection = Detection2D()

                box = b.xywh[0]

                # get boxes values
                detection.bbox.center.position.x = float(box[0])
                detection.bbox.center.position.y = float(box[1])
                detection.bbox.size_x = float(box[2])
                detection.bbox.size_y = float(box[3])

                # get track id
                track_id = ""
                if not b.id is None:
                    track_id = str(int(b.id))
                detection.id = track_id

                # get hypothesis
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = label
                hypothesis.hypothesis.score = score
                detection.results.append(hypothesis)

                # draw boxes for debug
                if label not in self._class_to_color:
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    self._class_to_color[label] = (r, g, b)
                color = self._class_to_color[label]

                min_pt = (round(detection.bbox.center.position.x - detection.bbox.size_x / 2.0),
                        round(detection.bbox.center.position.y - detection.bbox.size_y / 2.0))
                max_pt = (round(detection.bbox.center.position.x + detection.bbox.size_x / 2.0),
                        round(detection.bbox.center.position.y + detection.bbox.size_y / 2.0))
                cv2.rectangle(cv_image, min_pt, max_pt, color, 2)

                label = "{} ({}) ({:.3f})".format(label, str(track_id), score)
                pos = (min_pt[0] + 5, min_pt[1] + 25)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(cv_image, label, pos, font,
                            1, color, 1, cv2.LINE_AA)

                # append msg
                detections_msg.detections.append(detection)

            # publish detections and dbg image
            self._pub.publish(detections_msg)

            self._dbg_pub.publish(self.cv_bridge.cv2_to_imgmsg(cv_image,
                                                               encoding=msg.encoding))


def main():
    rclpy.init()
    node = Yolov8Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
