"""
WebSocket endpoint for live plan execution updates.

Provides real-time task status updates during plan execution.
"""

import asyncio
from typing import List, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends

from ..dependencies import get_bridge, GRPCBridge

router = APIRouter()


class ConnectionManager:
    """
    Manage WebSocket connections for live execution updates.
    
    Tracks active connections per plan_id and handles broadcasting
    updates to all connected clients.
    """
    
    def __init__(self):
        self.active_connections: Dict[int, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, plan_id: int):
        """Accept a new WebSocket connection for a plan."""
        await websocket.accept()
        if plan_id not in self.active_connections:
            self.active_connections[plan_id] = []
        self.active_connections[plan_id].append(websocket)
    
    def disconnect(self, websocket: WebSocket, plan_id: int):
        """Remove a WebSocket connection."""
        if plan_id in self.active_connections:
            self.active_connections[plan_id].remove(websocket)
            if not self.active_connections[plan_id]:
                del self.active_connections[plan_id]
    
    async def broadcast(self, plan_id: int, message: dict):
        """Send a message to all connections watching a plan."""
        if plan_id in self.active_connections:
            for connection in self.active_connections[plan_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass  # Connection may have closed


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws/execution/{plan_id}")
async def websocket_execution(websocket: WebSocket, plan_id: int):
    """
    WebSocket endpoint for live plan execution updates.

    Connects to receive real-time task status updates for a specific plan.
    Updates are pushed every second while connected.
    """
    # Get bridge without Depends (WebSocket doesn't support it the same way)
    from ..dependencies import get_bridge
    bridge = get_bridge()

    await manager.connect(websocket, plan_id)
    try:
        while True:
            # Poll for task updates
            tasks = bridge.list_tasks(plan_ids=[plan_id])
            await websocket.send_json({
                "type": "tasks_update",
                "plan_id": plan_id,
                "tasks": [t if isinstance(t, dict) else t.dict() for t in tasks]
            })
            await asyncio.sleep(1)  # Poll every second
    except WebSocketDisconnect:
        manager.disconnect(websocket, plan_id)


@router.websocket("/ws/global-updates")
async def websocket_global_updates(websocket: WebSocket):
    """
    WebSocket endpoint for global real-time updates.

    Sends periodic invalidation signals to trigger React Query refetches.
    This eliminates polling while providing real-time updates.
    """
    await websocket.accept()
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "message": "Real-time updates enabled"
        })

        while True:
            # Send invalidation signals every 3 seconds
            # This tells the frontend to refetch data if needed
            await websocket.send_json({
                "type": "invalidate",
                "queries": ["robot-health", "plans", "robots"],
                "timestamp": asyncio.get_event_loop().time()
            })
            await asyncio.sleep(3)  # Trigger updates every 3 seconds

    except WebSocketDisconnect:
        pass  # Connection closed
