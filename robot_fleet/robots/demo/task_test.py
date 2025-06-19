"""Test script for sending tasks to the FastAPI robot server"""

import sys
import json
import argparse
import requests

DEFAULT_SERVER = "http://0.0.0.0:5001/do_task"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Send a task to the HSR robot via FastAPI')
    parser.add_argument('task', type=str, help='Task description to send to the robot')
    parser.add_argument('--server', type=str, default=DEFAULT_SERVER, help='URL of the robot FastAPI server endpoint')
    parser.add_argument('--params', type=str, default='{}', help='Optional JSON string of parameters for the task')
    args = parser.parse_args()

    try:
        parameters = json.loads(args.params)
    except Exception as e:
        print(f"Error parsing parameters: {e}")
        sys.exit(1)

    payload = {
        "task_description": args.task,
        "parameters": parameters
    }

    print(f"Sending POST to {args.server} with payload:\n{json.dumps(payload, indent=2)}")
    try:
        response = requests.post(args.server, json=payload)
        response.raise_for_status()
        print("\nResponse:")
        print(json.dumps(response.json(), indent=2))
    except requests.RequestException as e:
        print(f"\nError communicating with robot server: {e}")
        if e.response is not None:
            print(f"Server response: {e.response.text}")
        sys.exit(1)
