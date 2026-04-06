from __future__ import annotations

import json
from typing import Callable, Optional

import websockets

from .events import GatewayRealtimeMessage


class GatewayRealtimeClient:
    def __init__(self, ws_base_url: str = "ws://localhost:8000"):
        self.ws_base_url = ws_base_url.rstrip("/")

    async def watch_global_updates(self, on_message: Callable[[GatewayRealtimeMessage], None]) -> None:
        url = f"{self.ws_base_url}/ws/global-updates"
        async with websockets.connect(url) as ws:
            async for msg in ws:
                try:
                    on_message(json.loads(msg))
                except Exception:
                    pass

    async def watch_plan_execution(self, plan_id: int, on_message: Callable[[GatewayRealtimeMessage], None]) -> None:
        url = f"{self.ws_base_url}/ws/execution/{plan_id}"
        async with websockets.connect(url) as ws:
            async for msg in ws:
                try:
                    on_message(json.loads(msg))
                except Exception:
                    pass

