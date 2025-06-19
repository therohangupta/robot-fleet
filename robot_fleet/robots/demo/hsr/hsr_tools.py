import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from tf_transformations import quaternion_from_euler
import argparse
import time
from typing import List
from hsrb_interface import Robot
from telemoma.robot_interface.hsr.hsr_core import HSR
import cv2
from datetime import datetime
from telemoma.utils.camera_utils import Camera
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import math
from demo.locations import locations

locations = locations["hsr"] 


TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigates the HSR to a specific (x, y) coordinate and orientation (yaw) on the map. Returns a success status boolean and a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "X coordinate in the map frame"},
                    "y": {"type": "number", "description": "Y coordinate in the map frame"},
                    "yaw": {"type": "number", "description": "Yaw orientation in radians"}
                },
                "required": ["x", "y", "yaw"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate_loc",
            "description": "Navigates the HSR to a named predefined location (e.g., 'kitchen', 'hallway'). Returns a success status boolean and a message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the location to navigate to. Must be one of: 'hallway', 'kitchen', 'living_room'"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reset_pose",
            "description": "Resets the HSR's arm, head, and gripper to a default pose. Returns a success status boolean and a message.",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Captures an image from the robot’s camera. Returns TWO PARAMETERS success status and image file path.",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segment_for_objects",
            "description": """
            Detects and returns a list of visible objects using YOLO v8, which segments for simple object labels. 
            The object labels returned by YOLO v8 segmentation are simple and do not include ownership or numeric information. 
            For example, if 3 cups are in the image, it will return an array of ['cup', 'cup', 'cup']. 
            Objects will not be labeled by who owns them. They will simply just be the object name. Search for the simplest labels you can.  
            Returns a success status boolean and a list of object names.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "The filepath of the image to segment. This is returned from capture_image()."
                    }
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": "Waits for user teleoperation to pick up a named object. Blocks until a flag file is created. Returns a success status boolean and a message.",
            "parameters": {
            "type": "object",
            "properties": {
                "object": {
                "type": "string",
                "description": "The name of the object to pick."
                }
            },
            "required": ["object"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": "Waits for user teleoperation to place a held object at a specified location or coordinates. Blocks until a flag file is created. Returns a success status boolean and a message.",
            "parameters": {
            "type": "object",
            "properties": {
                "object": {
                "type": "string",
                "description": "The name of the object to place."
                },
                "location": {
                "type": "string",
                "description": "The name of the target location (optional)."
                },
                "x": {
                "type": "number",
                "description": "Optional x coordinate to place the object."
                },
                "y": {
                "type": "number",
                "description": "Optional y coordinate to place the object."
                }
            },
            "required": ["object"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rotate_quarter_turn",
            "description": """
            Explicitly rotates the robot in place 90 degrees clockwise. 
            When exploring a location to find objects, it is a good idea should rotate the robot to different orientations a reasonable number of times it so that it sees a location from a variety of viewpoints. 
            Returns a success status boolean and a message.""",
            "parameters": {}
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log",
            "description": "Logs a string message to a timestamped file for later review. Void function, does not return anything.",
            "parameters": {
            "type": "object",
            "properties": {
                "statement": {
                "type": "string",
                "description": "The log message to save"
                }
            },
            "required": ["statement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "return_status",
            "description": "You return this every time you want to return a value from main().",
            "parameters": {
            "type": "object",
            "properties": {
                "task_success": {
                "type": "boolean",
                "description": "True if the robot completes its task successfully, and False when it does not."
                },
                "msg": {
                    "type": "string",
                    "description": """
                    A useful statement to describe the end state of the robot's task. 
                    If there is anything useful that the central planner that sends you the task should know, include it in this. 
                    This message will be directly injected into further replanning steps for future tasks."""
                },
                "replan_flag": {
                    "type": "boolean",
                    "description": """
                    Set this to True if the 'msg' parameter you passed in indicates that new information has been found in the environment that previously wasn't known.
                    This usually does not need to be set to True unless the task you are solving specifically requires discovering new information.
                    Replanning is required for most failures but not all failures. Specifically, if the robot fails to detect the target object or objects, no replanning is required and you set the replan_flag to False.
                    Only if a target object is found in the environment will it be considered new information and require replanning and the replan_flag to be set to True. 
                    """
                }
            },
            "required": ["task_success", "msg", "replan_flag"]
            }
        }
    }
]

TOOL_IMPORT_STRING = "from hsr_tools import return_status, place, pick, capture_image, segment_for_objects, reset_pose, navigate_loc, navigate, rotate_quarter_turn, log"

rclpy.init()
node = rclpy.create_node('hsr_tools')
hsr = HSR(node)

class NavGoalSender(Node):
    def __init__(self):
        super().__init__('nav_goal_sender')
        self._action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def build_goal(self, x: float, y: float, yaw_rad: float) -> NavigateToPose.Goal:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, yaw_rad)
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q

        goal = NavigateToPose.Goal()
        goal.pose = pose
        return goal
    
    def send_goal(self, x: float, y: float, yaw_rad: float):
        self.get_logger().info('Waiting for NavigateToPose action server...')
        self._action_client.wait_for_server()

        goal_msg = self.build_goal(x, y, yaw_rad)
        self.get_logger().info(f'Sending goal: x={x:.2f}, y={y:.2f}, yaw={yaw_rad:.2f} rad')

        send_goal_future = self._action_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal was rejected by the server.')
            return False, "Goal was rejected."

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        result_wrapper = get_result_future.result()
        status = result_wrapper.status
        result = result_wrapper.result

        status_msg_map = {
            0: "Unknown status.",
            1: "Goal accepted, waiting to execute.",
            2: "Goal executing.",
            3: "Goal canceling.",
            4: "Goal succeeded.",
            5: "Goal canceled.",
            6: "Goal aborted.",
        }

        status_msg = status_msg_map.get(status, f"Unexpected status code: {status}")
        if status == 4:
            self.get_logger().info(status_msg)
            return True, status_msg
        else:
            self.get_logger().error(status_msg)
            return False, status_msg


def return_status(task_success, msg, replan_flag):
    # node.destroy_node()
    return (task_success, msg, replan_flag)
    
def rotate_quarter_turn():
    try: 
        if not rclpy.ok():
            rclpy.init()

        pub = node.create_publisher(Twist, '/omni_base_controller/cmd_vel', 10)

        twist = Twist()
        twist.angular.z = np.pi / 8  # ~22.5°/s
        duration = 4  # seconds to make 90° turn at 22.5°/s

        node.get_logger().info("Rotating 90 degrees clockwise...")
        start_time = time.time()
        while time.time() - start_time < duration:
            pub.publish(twist)
            time.sleep(0.1)  # publish at 10 Hz

        # Stop motion
        stop_twist = Twist()
        pub.publish(stop_twist)

        node.get_logger().info("Rotation complete.")
    finally:
        return True, "Rotated 90 degrees clockwise"

def navigate_loc(location: str):
    if not rclpy.ok():
        rclpy.init()
    x, y, yaw = locations[location]
    return navigate(x, y, yaw)  # Return the result

def navigate(x: float, y: float, yaw: float):
    try:
        if not rclpy.ok():
            rclpy.init()
        nav_node = NavGoalSender()
        success, message = nav_node.send_goal(x, y, yaw)
        # node = rclpy.create_node('reset_gripper_node')
        reset_arm = {
            "arm": {
                'arm_flex_joint': 0.0,
                'arm_lift_joint': 0.0,
                'arm_roll_joint': 0.0,
                'wrist_flex_joint': -1.57,
                'wrist_roll_joint': 0.0
            }
        }
        hsr.whole_body.move_to_joint_positions(reset_arm["arm"])
    finally:
        return success, message  # Return status and message

def reset_pose():
    try:
        if not rclpy.ok():
                rclpy.init()
        

        reset_pose = {
            "arm": {'arm_flex_joint': 0.0,
                    'arm_lift_joint': 0.0,
                    'arm_roll_joint': 0.0,
                    'wrist_flex_joint': -1.57,
                    'wrist_roll_joint': 0.0},
            "head": {'head_pan_joint': 0.0,
                    'head_tilt_joint': -0.2},
            "gripper": {'hand_motor_joint': 1.0}
        }

        hsr.whole_body.move_to_joint_positions(reset_pose["arm"])
        hsr.whole_body.move_to_joint_positions(reset_pose["head"])
    finally:
        return True, "Gripper and arm reset to default pose"

from ultralytics import YOLO

def segment_for_objects(image_path: str) -> List[str]:
    # Load model
    filepath = image_path
    model = YOLO("yolov8n-seg.pt")  # Make sure the model file exists

    # Load image
    image = cv2.imread(filepath)
    if image is None:
        raise ValueError(f"Could not read image from {filepath}")

    # Run segmentation inference
    results = model(image)

    # Collect object names
    detected_objects = []

    for r in results:
        for i in range(len(r.boxes)):
            cls_id = int(r.boxes.cls[i])
            name = model.names[cls_id]
            conf = float(r.boxes.conf[i])
            print(f"Detected: {name} with confidence {conf:.2f}")
            detected_objects.append(name)

    # Save visualization
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
    os.makedirs(images_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filepath = os.path.join(images_dir, f"segmented_{timestamp}.jpg")
    results[0].save(filename=out_filepath)
    print(f"Saved segmented image to {out_filepath}")

    return True, detected_objects

def log(statement: str):
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    log_file_path = os.path.join(logs_dir, f"logs.txt")
    with open(log_file_path, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] {statement}\n")  

def capture_image(camera_name='head_camera'):
    try:
        if not rclpy.ok():
            rclpy.init()

        color_topic = "/head_rgbd_sensor/rgb/image_rect_color" if camera_name == "head_camera" else "/hand_camera/image_raw"
        depth_topic = "/head_rgbd_sensor/depth_registered/image_raw" if camera_name == "head_camera" else "/hand_camera/image_raw"

        camera = Camera(node, color_topic, depth_topic)
        image = camera.get_img()  # NumPy array in RGB

        if camera_name == "head_camera":
            image = image[..., [2, 1, 0]] # swaps the R and B color channels (for some reason we need this)

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)  # Safe conversion

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        if not os.path.exists(images_dir):
            os.mkdir(images_dir)
        images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
        filepath = os.path.join(images_dir, f"{camera_name}_{timestamp}.jpg")    
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"Saved image to {filepath}")
    finally:
        return True, filepath


import time

def pick(object):
    flag_path = "/tmp/hsr_done.txt"
    print(f"[Pick] Teleop now. Waiting for confirmation via: {flag_path}")
    while not os.path.exists(flag_path):
        time.sleep(1)
    os.remove(flag_path)
    return True, f"Picked up '{object}' manually"

def place(object, location=None, x=None, y=None):
    flag_path = "/tmp/hsr_done.txt"
    target = location or f"coordinates ({x:.2f}, {y:.2f})" if x and y else "an unspecified location"
    print(f"[Place] Teleop now. Waiting for confirmation via: {flag_path}")
    while not os.path.exists(flag_path):
        time.sleep(1)
    os.remove(flag_path)
    return True, f"Placed '{object}' at {target} manually"

 


def main():
    parser = argparse.ArgumentParser(description='HSR tools')
    subparsers = parser.add_subparsers(dest='command', required=True)

    nav_parser = subparsers.add_parser('navigate')
    nav_parser.add_argument('--x', type=float, required=True)
    nav_parser.add_argument('--y', type=float, required=True)
    nav_parser.add_argument('--yaw', type=float, required=True)

    explore_parser = subparsers.add_parser('navigate_loc')
    explore_parser.add_argument('--location', type=str, required=True, help="location to nav to")

    explore_parser = subparsers.add_parser('explore')
    explore_parser.add_argument('--locations', type=str, nargs='+', required=True, help="List of location names to visit")

    subparsers.add_parser('reset_gripper')

    args = parser.parse_args()

    if args.command == 'navigate':
        navigate(args.x, args.y, args.yaw)
    elif args.command == 'navigate_loc':
        navigate_loc(args.location)
    elif args.command == 'explore':
        explore(args.locations)
    elif args.command == 'reset_gripper':
        reset_pose()

if __name__ == '__main__':
    main()