import sys
import rospy
from geometry_msgs.msg import PoseStamped, Twist
# from nav2_msgs.action import NavigateToPose
# from tf_transformations import quaternion_from_euler
import argparse
import time
from typing import List
sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/')
from interbotix_xs_modules.locobot import InterbotixLocobotXS
import tf
import os
import numpy as np
import math
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
from datetime import datetime
from ultralytics import YOLO
from demo.locations import locations

locations = locations["locobot"] 

locobot = InterbotixLocobotXS(robot_model="locobot_base", use_move_base_action=True)

TOOL_DESCRIPTIONS = [
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigates the Locobot to a specific (x, y) coordinate and orientation (yaw) on the map. Returns a success status boolean and a string message.",
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
            "description": """
            Navigates the Locobot to a named predefined location (e.g., 'kitchen', 'hallway') and sets it to a predetermined orientation. 
            This just goes to a location but DOES NOT explore it. To explore a location, use the other functions. 
            Returns a success status boolean and a message.""",
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
            "description": "Resets the Locobot's camera to its original position looking straight ahead. Returns a success status boolean and a message.",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "camera_look",
            "description": """
Explicitly only changes the Locobot's camera to look at a certain direction. 
Use this to explore and find objects in a specific location from a specific camera angle. You need to explore using all camera directions if asked to find an object or objects.
Looking in all camera angles after arriving at a certain location will give the best information about objects in a scene.
Returns a success status boolean and a message.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "number",
                        "description": """The direction the camera will be set to. Must be an integer between 1 and 5, inclusive. 1 looks up at 60 degrees, 2 looks up at 45 degrees, 3 looks straight ahead, 4 looks down at 45 degrees, and 5 looks down at 60 degrees."""
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "capture_image",
            "description": "Captures an image from the robot's current camera position and angle. Returns TWO PARAMETERS success status and image file path.",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_found_object_string",
            "description": "Creates an easy-to-read string about what objects are found at a specific location in the environment. Only use this when the objects array you pass has at least 1 item. Returns ONE VALUE: a string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The name of the location where objects are found. Must be one of: 'hallway', 'kitchen', 'living_room'."
                    },
                    "objects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of objects that were detected to be found in the location."
                    }
                },
                "required": ["location", "objects"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segment_for_objects",
            "description": "Detects and returns a list of visible objects for a single given image taken by the robot at a specific camera angle and position. Returns a success status boolean and a list of object names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Filepath of the image to segment. This is returned from capture_image()."
                    }
                },
                "required": ["filepath"]
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
            "name": "get_current_location",
            "description": "Gets the robot's current location in x, y, and yaw as a tuple (x, y, yaw). Returns a success status boolean followed by the location 3-tuple as follows: (bool, 3-tuple).",
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
            "task_success_status": {
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
          "required": ["task_success_status", "msg", "replan_flag"]
        }
      }
    }
]

TOOL_IMPORT_STRING = "from locobot_tools import return_status, get_current_location, capture_image, camera_look, create_found_object_string, segment_for_objects, reset_pose, navigate_loc, navigate, rotate_quarter_turn, log"
        
def rotate_quarter_turn():
    locobot.base.move(0.1, -math.pi/2, 1.2)

    return True, "Rotated 90 degrees clockwise"

def navigate_loc(location: str):
    camera_look(3)
    x = locations[location][0]
    y = locations[location][1]
    yaw = locations[location][2]
    return locobot.base.move_to_pose(x, y, yaw, wait=True), f"Reached {location}"

def navigate(x: float, y: float, yaw: float):
    camera_look(3)
    nav = locobot.base.move_to_pose(x, y, yaw, wait=True)
    if nav:
        return True, f"Navigated to ({x}, {y}) with orientation {yaw}"
    else:
        return False, "Failed to navigate to desired location"

def reset_pose():
    locobot.camera.pan_tilt_go_home()

    return True, "Camera reset to default pose"

def camera_look(direction): 
    if direction == 1:
        locobot.camera.tilt(-math.pi/3)
    elif direction == 2:
        locobot.camera.tilt(-math.pi/4)
    elif direction == 3:
        locobot.camera.pan_tilt_go_home()
    elif direction == 4:
        locobot.camera.tilt(math.pi/4)
    elif direction == 5:
        locobot.camera.tilt(math.pi/3)

    return True, f"Camera to position {direction}"

def segment_for_objects(filepath: str) -> List[str]:
    # Load model
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
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp/yolo/")
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

def capture_image(topic="/locobot/camera/color/image_raw", timeout=5.0):
    # topic="/locobot/camera/color/image_raw"
    bridge = CvBridge()
    now_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_path="./tmp/"+now_str+".png"
    try:
        msg = rospy.wait_for_message(topic, Image, timeout=5.0)
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        cv2.imwrite(output_path, cv_image)
        rospy.loginfo(f"Image saved to {output_path}")
        return True, output_path
    except rospy.ROSException:
        raise TimeoutError(f"No image received on topic '{topic}' within {timeout} seconds.")
    except Exception as e:
        rospy.logerr(f"Image saving failed: {e}")
        raise


def create_found_object_string(location, objects):
    foundObjs = f"Found the following objects in {location}: " + ", ".join(objects)

    return foundObjs

def return_status(task_success_status, msg, replan_flag):
    return (task_success_status, msg, replan_flag)
    
def get_current_location():
    """
    Returns the robot's current pose in the 'map' frame as (x, y, yaw).
    Returns None if the transform is unavailable.
    """
    listener = tf.TransformListener()
    try:
        listener.waitForTransform("map", "locobot/base_footprint", rospy.Time(0), rospy.Duration(5.0))
        (trans, rot) = listener.lookupTransform("map", "locobot/base_footprint", rospy.Time(0))
        x, y = trans[0], trans[1]
        _, _, yaw = tf.transformations.euler_from_quaternion(rot)
        return True, (round(x, 5), round(y, 5), round(yaw, 5))
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
        rospy.logwarn("Failed to get transform map -> base_link: %s", str(e))
        return None



def main():
    parser = argparse.ArgumentParser(description='Locobot tools')
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