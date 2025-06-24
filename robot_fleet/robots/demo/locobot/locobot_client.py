import os
import sys
import time
import threading
import requests


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

# pip install openai fastapi dotenv uvicorn ultralytics
# To run: uvicorn locobot_server:app --host 0.0.0.0 --port 5001