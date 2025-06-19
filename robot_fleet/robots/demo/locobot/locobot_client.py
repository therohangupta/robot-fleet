import os
import sys
import time
import threading
import requests

STATUS_PATH = "/tmp/locobot_status.txt"
DONE_PATH = "/tmp/locobot_done.txt"

def main():
    if len(sys.argv) < 2:
        print("Usage: locobot 'task description'")
        sys.exit(1)

    task_description = sys.argv[1]
    url = "http://localhost:5001/do_task"
    payload = {
        "task_description": task_description
    }

    print(f"🚀 Sending task: {task_description}")

    # Send request to robot API
    try:
        response = requests.post(url, json=payload)
        print("\n📬 Task result:")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Failed to send request: {e}")

if __name__ == "__main__":
    main()

# alias hsr="python3 /root/TEMP/telemoma/telemoma/robot_interface/robot_fleet_server/hsr_client.py"