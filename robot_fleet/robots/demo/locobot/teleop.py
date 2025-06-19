# REQUIRED IMPORTS #
import gzip
import json
from gym import spaces
from importlib import import_module
import torch
import torchvision
import cv2
import time
import math
import os
import datetime
import sys
import numpy as np
import rospy
from PIL import Image as PILIMAGE

sys.path.append('/home/elle/interbotix_ws/src/interbotix_ros_toolboxes/interbotix_xs_toolbox/interbotix_xs_modules/src/')
from interbotix_xs_modules.locobot import InterbotixLocobotXS

# SETUP OF LOCOBOT AND GLOBAL VARS

locobot = InterbotixLocobotXS(robot_model="locobot_base")
locobot.camera.pan_tilt_go_home()
print("robot setup")

# USER-CONTROLLED ACTIONS
def do_action(action, locobot):
    if locobot == None:
        return
    if action == 'q':
        print("stop")
        # import pdb; pdb.set_trace()
    elif action == 'w':
        locobot.base.move(0.25, 0, 2.0)
        print('moved forward')
    elif action[0] == 'w' and len(action)>0:
        locobot.base.move(0.25, 0, int(action[1:])*1.0)
        print('moved forward')
    elif action == 'a':
        locobot.base.move(0.1, math.pi / 6.0, 1.3)
        print('turned left')
    elif action == 'n':
        locobot.base.move(0.1, math.pi / 3.0, 1.3)
        print('turned left')
    elif action == 'd':
        locobot.base.move(0.1, -math.pi / 6.0, 1.2)
        print('turned right')
    elif action == 'm':
        locobot.base.move(0.1, -math.pi / 3.0, 1.2)
        print('turned right')
    elif action == 's':
        locobot.camera.tilt(math.pi / 12.0)
        print('looked down')
    elif action == 'e':
        locobot.camera.tilt(-math.pi / 12.0)
        print('looked up')
    elif action == 'x':
        locobot.camera.pan_tilt_go_home()
        print('looked straight')
    elif action == 'z':
        for i in range(10):
            locobot.base.move(0.1, -math.pi / 3.0, 1.2)
    time.sleep(1)
    return 
    # return get_observation(locobot)

# Returns next episode number based on existing episodes in "episodes" directory
def get_next_episode_number():
    folder_path = "episodes"
    # Get the list of existing folders in the directory
    existing_folders = os.listdir(folder_path)
    existing_folders = [int(folder.split("_")[1]) for folder in existing_folders]

    # Sort the folders in descending order
    existing_folders.sort(reverse=True)
    print(existing_folders)
    # # Check if any folder exists
    if existing_folders:
        # Get the latest folder number
        episode_number = existing_folders[0]

        # Increment the episode number by 1
        new_episode_number = episode_number + 1
        print("new ep = ", new_episode_number)
        # Create the new folder with the incremented episode number
        new_folder = f"episode_{new_episode_number}"
        new_folder_path = os.path.join(folder_path, new_folder)
        os.makedirs(new_folder_path)
        return new_folder
    else:
        # If no folder exists, create the first folder
        new_folder = "episode_0"
        new_folder_path = os.path.join(folder_path, new_folder)
        os.makedirs(new_folder_path)
        return new_folder

def getImageNumber(folder_path, img_type):
    # Get the list of existing images in the directory
    existing_images = os.listdir(folder_path)
    print(existing_images)
    # Sort the images in descending order
    existing_images.sort(reverse=True)

    # # Check if any folder exists
    if existing_images:
        # Get the latest image name
        latest_image = existing_images[0]

        # Extract the episode number from the latest folder name
        image_number = int(latest_image.split(img_type[len(img_type)-1])[1].split(".")[0])

        # Increment the episode number by 1
        new_image_number = image_number + 1

        # Create the new image with the incremented episode number
        new_image = img_type + str(new_image_number).zfill(4)
        return new_image
    else:
        # If no image exists, create the first folder
        new_image = img_type+"0000"
        return new_image

# Collects the RGB and Depth images and places them in the current episode's folder
def collectImages(folder_path):
    color_image = None
    depth_image = None
    if locobot != None:
        color_image, depth_image = locobot.base.get_img()
        depth_image = depth_image.astype(np.float32)
        # print(type(depth_image[0][0]))
        # depth_image = depth_image * 10.0
        # print(type(depth_image[0][0]))
        # depth_image[depth_image > 65535.0] = 65535.0
        # print(depth_image)
        # max = depth_image.max()
        # mini = depth_image.min()
        # print("max, ", max)
        # print("min, ", mini)
        color_img = PILIMAGE.fromarray(color_image.astype(np.uint8))
        depth_img = PILIMAGE.fromarray(depth_image.astype(np.uint16), mode='I;16')
        color_path = getImageNumber(folder_path + "/rgb/", "rgb") + ".png"
        color_img.save(folder_path + "/rgb/" + color_path)
        depth_path = getImageNumber(folder_path + "/depth/", "depth") + ".png"
        depth_img.save(folder_path + "/depth/" + depth_path)



def data_collection():
    possible_actions = ["x", "w", "a", "d", "e", "s"]
    action_names = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "LOOK_UP", "LOOK_DOWN"]
    action_path_num = []
    action_path_name = []
    folder_path = "./episodes/"
    json_ = None
    curr_folder = folder_path + get_next_episode_number()
     # changed for each episode
    instruction_text = "Turn right and you will see a bookshelf. Go to the bookshelf, and then turn left and you will see a table. Go to the table and then turn left and you will see a bed. Go to the bed and then stop."
    print(curr_folder)
    
    # saving initial images to curr_folder
    rgb_folder_path = os.path.join(curr_folder, "rgb")
    os.makedirs(rgb_folder_path)
    depth_folder_path = os.path.join(curr_folder, "depth")
    os.makedirs(depth_folder_path)
    collectImages(curr_folder)
    totalDistance = 0
    start_time = time.time()
    while(1):
        cmd = input("What do you want to do?\nw = move_fwd, a = turn_left 30 deg, d = turn_right 30 deg, e = look_up, s = look_down, x = stop, q = quit\n")
        if(cmd == 'q'): # stops the episode due to error
            break
        if(cmd == 'w'):
            totalDistance = totalDistance + 0.25
        do_action(cmd, locobot)
        
        act_ind = possible_actions.index(cmd)
        action_path_num.append(act_ind)
        action_path_name.append(action_names[act_ind])
        if(cmd == 'x'): # stops when the goal is reached
            # save dictionary as json
            end_time = time.time() - start_time
            json_ = {
                "instruction": instruction_text,
                "actions": action_path_num,
                "action_names": action_path_name,
                "distance_travelled": totalDistance,
                "time_taken": end_time
            }
            json_object = json.dumps(json_, indent = 4)
            episode_json = os.path.join(curr_folder, "episode.json")
            with open(episode_json, "w") as file:
                file.write(json_object)
            break
        collectImages(curr_folder)
        

def regular_teleop():
    while(1):
        cmd = input("What do you want to do?\nw = move_fwd, a = turn_left 30 deg, d = turn_right 30 deg\n")
        do_action(cmd, locobot)
        if(cmd == 'q'):
            break


def main(): 
    print(sys.argv[1])
    if(sys.argv[1] == '-reg'):
        regular_teleop()
    elif(sys.argv[1] == '-collect'):
        data_collection()

if __name__ == "__main__":
    main()