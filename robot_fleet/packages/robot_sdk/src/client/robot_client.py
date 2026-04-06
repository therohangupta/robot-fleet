import httpx
import logging

from ..models import TaskResult


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RobotClient:
    def __init__(self, host: str, port: int):
        if not host or not isinstance(port, int):
            raise ValueError("Host and port must be provided and valid.")
        self.base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(timeout=None)
        logger.info(f"RobotClient initialized for server at {self.base_url}")

    async def do_task(self, task_description: str) -> TaskResult:
        endpoint = f"{self.base_url}/do_task"
        payload = {"task_description": task_description}

        logger.info(f"[RobotClient] Sending task: {payload} to {endpoint}")

        try:
            response = await self._client.post(endpoint, json=payload)
            logger.info(f"[RobotClient] Response received: {response.status_code}")
            response.raise_for_status()
            response_data = response.json()
            logger.info(f"[RobotClient] Response data: {response_data}")
            return TaskResult(**response_data)

        except httpx.HTTPStatusError as e:
            logger.error(f"[RobotClient] HTTP error: {e.response.status_code} - {e.response.text}")
            raise Exception(f"HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"[RobotClient] Request error: {e}")
            raise Exception(f"Request error: {str(e)}")
        except Exception as e:
            logger.error(f"[RobotClient] Unexpected error: {e}", exc_info=True)
            raise

    async def close(self):
        await self._client.aclose()
        logger.info("RobotClient connection closed.")

