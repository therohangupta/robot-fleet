import os
import sys
import time
import threading
import requests

STATUS_PATH = "/tmp/hsr_status.txt"
DONE_PATH = "/tmp/hsr_done.txt"

def watch_for_user_prompts():
    last_status = None
    while True:
        try:
            if os.path.exists(STATUS_PATH):
                with open(STATUS_PATH, 'r') as f:
                    status = f.read().strip()

                if status != last_status and status in ["ready to pick", "ready to place"]:
                    print(f"\n🤖 HSR is {status.upper()} — press ENTER when done.")
                    input()
                    with open(DONE_PATH, 'w') as f:
                        f.write("done")
                    print("✅ Confirmation sent.\n")
                    last_status = status
            else:
                last_status = None
            time.sleep(0.5)
        except Exception as e:
            print(f"[Watcher Error] {e}")
            break

def main():
    if len(sys.argv) < 2:
        print("Usage: hsr 'task description'")
        sys.exit(1)

    task_description = sys.argv[1]
    url = "http://localhost:5001/do_task"
    payload = {
        "task_description": task_description
    }

    print(f"🚀 Sending task: {task_description}")

    # Start watcher in background thread
    watcher_thread = threading.Thread(target=watch_for_user_prompts, daemon=True)
    watcher_thread.start()

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