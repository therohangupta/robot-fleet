### LoCoBot Information
This README is for our information regarding the LoCoBot.

### Mapping and Navigation
This LoCoBot uses rtabmap to create a 2D map of its environment. The map is saved to /home/<user>/.ros/rtabmap.db, from which you can copy it into any directory to avoid it getting overwritten. This robot doesn't use lidar. 

The command to launch an existing map of the environment is: `roslaunch interbotix_xslocobot_nav xslocobot_nav.launch robot_model:=locobot_base use_lidar:=false localization:=true database_path:=<path to rtabmap file>`

The command to create a new map is: `roslaunch interbotix_xslocobot_nav xslocobot_nav.launch robot_model:=locobot_base use_lidar:=false rtabmap_args:=-d`. This deletes the pre-existing map saved to /home/elle/.ros/rtabmap.db.

Good practice is to copy the map you like to a maps directory that exists outside your workspace and name it something meaningful. 

More documentation for the LoCoBot Nav stack at: https://docs.trossenrobotics.com/interbotix_xslocobots_docs/ros1_packages/navigation_stack_configuration.html#overview.

If creating a new map, you will see its current state as you teleoperate it to new locations. Go slow and steady and make sure the robot sees all meaningful features and obstacles in the environment. 

### Teleoperation

Run `teleop.py -reg` to teleoperate this robot using W,A, and D. N and M turn the robot more while S tilts the camera down and E tilts it up. 

### Foxglove

If you have Foxglove installed, you can launch it on the robot with `roslaunch foxglove_bridge foxglove_bridge.launch`. This launches by default on port 8765. It is advised to launch this on a terminal not in any IDE, but rather the normal computer terminal. On Foxglove, you connect to the ip address and port as follows: `<ip-address-or-alias>:8765`. For example, if my robot's ip address alias is 'elle-NUC103iFNK.local', then I'd connect to Foxglove with 'ws://elle-NUC103iFNK.local:8765'.

### Common Issues

1. Local Costmap no longer updates
After launching the navigation stack, the local costmap topic `/locobot/move_base/local_costmap/costmap` may stop receiving updates. If this is the case, you will see a blank square around the robot instead of any black blobs representing detected obstacles. A functional local costmap that detects obstacles is crucial for proper navigation on the map. To fix this, you must restart the base by clicking and holding the power button with a white circle around it and wait till it turns off. Once turned off, the only way to turn it back on again is to put it on its charger. A light will flash and move in a circle but the base is only ready once it makes a noise and the light no longer moves in a circle but just stays white. After doing this, relaunch the navigation stack 

2. The map doesn't show up on Foxglove after launching the navigation stack
This isn't really an issue, as the map takes time to be loaded and won't recognize the map target_frame until the .db file has been completely loaded. You will see this warning: "[WARN] [1747627238.208641271]: Timed out waiting for transform from locobot/base_footprint to map to become available before running costmap, tf error: canTransform: target_frame map does not exist.. canTransform returned after 0.100371 timeout was 0.1." until the map is loaded. Be patient and just let the map load.

3. 



