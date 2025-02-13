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

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():

    #
    # ARGS
    #
    model = LaunchConfiguration("model")
    model_cmd = DeclareLaunchArgument(
        "model",
        default_value="yolov8n-pose.pt",
        description="Model name or path")

    tracker = LaunchConfiguration("tracker")
    tracker_cmd = DeclareLaunchArgument(
        "tracker",
        default_value="bytetrack.yaml",
        description="Tracker name or path")

    device = LaunchConfiguration("device")
    device_cmd = DeclareLaunchArgument(
        "device",
        default_value="cuda:0",
        description="Device to use (GPU/CPU)")

    enable = LaunchConfiguration("enable")
    enable_cmd = DeclareLaunchArgument(
        "enable",
        default_value="True",
        description="Wheter to start darknet enabled")

    threshold = LaunchConfiguration("threshold")
    threshold_cmd = DeclareLaunchArgument(
        "threshold",
        default_value="0.5",
        description="Minimum probability of a detection to be published")

    input_image_topic = LaunchConfiguration("input_image_topic")
    input_image_topic_cmd = DeclareLaunchArgument(
        "input_image_topic",
        default_value="/camera/rgb/image_raw",
        description="Name of the input image topic")

    pointcloud_topic = LaunchConfiguration("pointcloud_topic")
    pointcloud_topic_cmd = DeclareLaunchArgument(
        "pointcloud_topic",
        default_value="/camera/depth_registered/points",
        description="Name of the pointcloud topic")
    
    object_detections_topic = LaunchConfiguration("object_detections_topic")
    object_detections_topic_cmd = DeclareLaunchArgument(
        "object_detections_topic",
        default_value="/yolo/bags/detections",
        description="Topic name of the detected object to point")
    
    person_detections_topic = LaunchConfiguration("person_detections_topic")
    person_detections_topic_cmd = DeclareLaunchArgument(
        "person_detections_topic",
        default_value="/yolo/detections",
        description="Topic name of the detected object to point")

    namespace = LaunchConfiguration("namespace")
    namespace_cmd = DeclareLaunchArgument(
        "namespace",
        default_value="yolo",
        description="Namespace for the nodes")

    #
    # NODES
    #
    detector_node_cmd = Node(
        package="yolov8_ros",
        executable="yolov8_node",
        name="yolov8_node",
        namespace=namespace,
        parameters=[{"model": model,
                     "tracker": tracker,
                     "device": device,
                     "enable": enable,
                     "threshold": threshold}],
        remappings=[("image_raw", input_image_topic)]
    )

    detector3d_node_cmd = Node(
        package="yolov8_ros",
        executable="yolov8_3d_node",
        name="yolov8_3d_node",
        namespace=namespace,
        remappings=[("detections", object_detections_topic), 
                    ("person_detections", person_detections_topic),
                    ("depth_points", pointcloud_topic)]
    )

    ld = LaunchDescription()

    ld.add_action(model_cmd)
    ld.add_action(tracker_cmd)
    ld.add_action(device_cmd)
    ld.add_action(enable_cmd)
    ld.add_action(threshold_cmd)
    ld.add_action(input_image_topic_cmd)
    ld.add_action(namespace_cmd)
    ld.add_action(pointcloud_topic_cmd)
    ld.add_action(object_detections_topic_cmd)
    ld.add_action(person_detections_topic_cmd)
    ld.add_action(detector_node_cmd)
    ld.add_action(detector3d_node_cmd)

    return ld
