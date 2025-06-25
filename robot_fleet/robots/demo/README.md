# Understanding the Demo

In this section, we will explain our demo, showing with an example of a 2-robot fleet. We have a Toyota HSR and Locobot in our fleet. You can see our demo on both these robots here: https://youtu.be/fEF9vxRA-eU. 

## General Structure of Robot Directories
Every robot has the following files that allow it to work in our multi-robot task planning and allocation framework. Here is a description of each file

### YAML
This is in the form of `robot_name.yaml`. This file primarily serves to allow an easy configuration of the robot's natural language description, networking configurations (hosts, ports, firewalls if necessary), and Docker deployments if the robot is itself running on a Docker container. 

For example, in `hsr.yaml`, we describe our HSR as follows: 
```
metadata:
  name: HSR
  description: "Human support robot made by toyota"

deployment:
  docker_host: localhost  # Default to local deployment for testing
  docker_port: 2375

taskServer:
  host: hsrc.local
  port: 5001

container:
  image: robotfleet/pick-place:1.0
  environment:
    ROBOT_ID: ""  # Will be set when instantiating
    LOG_LEVEL: info
    PORT: "5003"  # Default port for the robot server
    ARM_CONFIG: ""  # Required for arm control
    GRIPPER_CONFIG: ""  # Required for gripper control
  required_devices:
    - /dev/ttyUSB*  # For arm and gripper

capabilities:
  - can navigate to known locations
  - can pick up objects
  - can place objects
  - can place objects on most navigation robots
  - can only hold one object at a time
  - can deliver objects it can hold and carry
  - can search for objects
```

The `metadata` has properties such as `name` and `description` which can be used but are more for your own information rather than anything RobotFleet may use.  
The `capabilities` are used by high level planners and allocators in LLM prompts. These can guide and assist the LLM with generating plans that make sense for the robots.

The only difference between the `fake_` and regular robot yaml files is the `taskServer.host` — the fake robot will be runnable on your localhost while the actual robot has a different host IP. 


Every robot has its own `[robot name]_client.py`, `[robot name]_server.py`, and `[robot name]_tools.py`. 

- `[robot name]_tools.py`: These include a set of functions that are defined as tools for an LLM. Each of these functions will encode a capability for the robot, such as `navigate`, `rotate`, `segment_for_objects`, `capture_image`, etc. that each implement the lowest level of code on each robot. These capabilities will have very similar type signatures and parameters across many embodiments. For example, the HSR's `navigate` function is as follows: 
    ```
    def navigate_loc(location: str):
        if not rclpy.ok():
            rclpy.init()
        x, y, yaw = locations[location]
        return navigate(x, y, yaw)  # Return the result
    ```
    whereas on the Locobot it is implemented as: 
    ```
    def navigate_loc(location: str):
        camera_look(3)
        x = locations[location][0]
        y = locations[location][1]
        yaw = locations[location][2]
        return locobot.base.move_to_pose(x, y, yaw, wait=True), f"Reached {location}"
    ```

    Both functions have the same parameters and name and description for the LLM, but they are each implented differently depending on the specific robot-specific APIs. The HSR was running ROS2 and the Locobot was running ROS1 — so these functions are higher level wrappers that create common capabilties across robots regardless of the underlying low-level tech stack. These simply become capabilities or skills that each of the robots have and that an LLM can understand in tool calling. 

- `[robot name]_client.py`: This is a simple testing file that you can use to test sending network requests where the payload is the task description to the robot's server. This is not used in any end-to-end run of the entire 2-robot fleet. For example, if you just want to send a test instruction to the HSR, you could just run `python hsr_client.py "go to the kitchen"` and similarly for the Locobot. In the end-to-end demo, a `RobotClient` is instantiated for every robot that establishes network requests. 
- `[robot name]_server.py`: This is where the magic starts to happen. This is where you implement your way of breaking the task description down to actions. Every robot server should extend the `RobotServerBase` class. There are two main functions: `execute_task` and `do_task`. In `_execute_task`, you implement how you want the robot to perform on the `task_description` it is given. This `task_description` can be augmented with more parameters or anything you'd like to pass from the robot's client. In our demo, we simply just take the `task_description`, which is simply just a natural language string, and breaks it down into code based on a two-step breakdown: step 1 is creating "pseudocode" where the robot uses its capabilities, as defined by the standard tool-calling functions and descriptions in the `[robot name]_tools.py`, and step 2 is turning that pseudocode into runnable Python script and performing some post-processing on it. We then run that code, which exists as a temp file, and it would include functions from `[robot name]_tools.py`. After your `execute_task` completes, you need to pass that result as a `TaskResult` message and return it. The `do_task` function will return the result of `execute_task` with some HTTP request error handling that can be implemented with the commented out code. 

`TaskResult` is a simple class that is made of three properties: 
- `success`: boolean representing if the task was successfully completed
- `message`: a natural language message saying if the robot succeeded or not and any other useful info about the task's success or failure
- `replan`: boolean indicating whether generating a new plan is required, such as in the case of a task failure or discovery of new information in the environment

A closer look at the functions in robots' `tools.py` file shows a `return_status` function that we tell the LLM to use instead of a standard variable as it allows easy return of values that can become the properties in the `TaskResult`. 

## Your Implementation

We provide a standard way to pass language tasks to robots in a fleet. Our demo used the single BigDAG DAG planning method along with LLM-powered assignment of robots to tasks, which works for our 2-robot fleet. If you want to add a new robot using our demo's planning and allocation configuration, you just create the robot's yaml, tools, and server. If you choose to use our pseudocode-to-script strategy, then you can just copy and change names in the prompts — the Locobot and HSR server prompts for both the pseudocode plan and code generation are not that different. You would also have to implement different tools based on your robot's own APIs for interacting with end effectors, mobile bases, camera, ROS modules, etc.

If you want to change how you implement your execute_task for a robot — maybe you have a better code generator that uses an end-to-end policy, then you can modify execute_task to use that while easily using `TaskResult` and other standard classes we have defined for the multi-robot coordination messages. Let's say for example, you just want `_execute_task` to run a policy with a language input. You can define some functions in `[robot name]_tools.py`, add those as imports to the server file, and then your `_execute_task` can have code like so:

```
from [robot name]_tools import run_policy

...

class RobotName(RobotServerBase):
...
    async def _execute_task(self, task_description: str) -> TaskResult:
        actual_task_description = task_description.split("DO THE FOLLOWING TASK:")[1].strip()
        # assuming it returns a single boolean of whether or not the policy successfully executed
        is_success = run_policy(actual_task_description)

        # "message" and "replan" effectively have some no-op message or standard message as this can be maintained by the central planner as action/task success history
        result = TaskResult(success=is_success, message="This task successfully executed", replan=False)

        return result
```

It's as simple as that — define your own functions in tools, import and use them within `_execute_task`it into the server file, and return it as a TaskResult. We hope to support more functions in `[robot name]_tools.py` that you can easily modify.


## Experiments

In the demo, we asked the fleet to achieve 2 goals: "bring the first cup to the hallway" and "bring the second cup to the living room", and we specified the world state as follows: 
- "The HSR is in an unknown location"
- "The Locobot is in an unknown location"
- "There is a kitchen, a living room and a hallway."
- "The kitchen has 2 cups."

Our world state is currently a 2D semantic map as defined in locations plus some text statements about the environment. 

If you look at `demo1_cli.txt`, we specify the above using the command line (see the main RobotFleet README.md for how that's used) and create a plan using our BigDAG planning strategy and LLM allocation strategy. The generated plan can be seen in `demo1_generated_plan.txt`. 

The way to interpret the Tasks (DAG View) triples:
- "G#": the first element in the triple which represents the goal a task is most relevant to solving
- "T#": the second element in the triple which represents a unique identifier for the task (just ascending order)
- robot-id: the third element in the triple, just a unique identifer for a robot in your fleet

Each task also will say what tasks it depends on, forming that DAG structure. The task description is the actual natural language instruction sent to that robot-id.

In `demo1_execution.txt`, we display the logs that the central planner received as it was maintaining the status of various robots in the fleet and their tasks. This demo matches with the demo video exactly, though the video paraphrases the instructions for brevity. Take a look and you will see how the DAG plan is executed and statuses are logged every few seconds. At the very end, it says `Plan 1 completed successfully`, indicating every task in the DAG successfully executed. 

In `hardcode_tasks.txt`, we show how to use the CLI to manually hardcode the plan's tasks. This is manual hardcoding, so no planning or allocation strategy is used.
