from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import httpx


@dataclass
class GatewayClient:
    """
    Minimal Python client for the Gateway REST API.

    Notes:
    - This is intentionally thin; it should mirror the gateway REST contract.
    - Add richer typed models as the contract stabilizes.
    """

    base_url: str = "http://localhost:8000"
    bearer_token: Optional[str] = None
    timeout_seconds: float = 30.0

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def _request(self, method: str, path: str, json: Any | None = None) -> Any:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout_seconds) as client:
            resp = client.request(method, url, json=json, headers=self._headers())
            if resp.status_code >= 400:
                try:
                    detail = resp.json().get("detail")
                except Exception:
                    detail = resp.text
                raise RuntimeError(detail or f"HTTP {resp.status_code}")
            return resp.json()

    # Plans
    def list_plans(self) -> Any:
        return self._request("GET", "/api/plans")

    def get_plan(self, plan_id: int) -> Any:
        return self._request("GET", f"/api/plans/{plan_id}")

    def create_plan(self, payload: dict[str, Any]) -> Any:
        return self._request("POST", "/api/plans", json=payload)

    def allocate_plan(self, plan_id: int, allocation_strategy: str) -> Any:
        return self._request("POST", f"/api/plans/{plan_id}/allocate", json={"allocation_strategy": allocation_strategy})

    def start_plan(self, plan_id: int) -> Any:
        return self._request("POST", f"/api/plans/{plan_id}/start")

    # Tasks
    def list_tasks(self, plan_id: Optional[int] = None) -> Any:
        q = f"?plan_id={plan_id}" if plan_id is not None else ""
        return self._request("GET", f"/api/tasks{q}")

    def update_task(self, task_id: int, payload: dict[str, Any]) -> Any:
        return self._request("PATCH", f"/api/tasks/{task_id}", json=payload)

    def delete_task(self, task_id: int) -> Any:
        return self._request("DELETE", f"/api/tasks/{task_id}")

    # Robots
    def list_robots(self, filter: str = "all") -> Any:
        return self._request("GET", f"/api/robots?filter={filter}")

    def register_robot(self, payload: dict[str, Any]) -> Any:
        return self._request("POST", "/api/robots/register", json=payload)

    def unregister_robot(self, robot_id: str) -> Any:
        return self._request("DELETE", f"/api/robots/{robot_id}")

