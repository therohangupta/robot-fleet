"""
In-memory heartbeat store.

Stores the last N heartbeats per robot and derives health summary.
Later this can be swapped for Redis or another persistent store without changing the API.
"""

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from .config import MAX_HEARTBEATS_PER_ROBOT, HEARTBEAT_REACHABLE_THRESHOLD_SECS


@dataclass
class Heartbeat:
    ts: float
    reachable: bool
    busy: Optional[bool] = None


@dataclass
class RobotHealthState:
    history: deque = field(default_factory=lambda: deque(maxlen=MAX_HEARTBEATS_PER_ROBOT))
    last_seen: float = 0.0
    last_reachable: bool = False
    last_busy: Optional[bool] = None


class HeartbeatStore:
    """
    Thread-safe in-memory store for robot heartbeats.

    - Stores the last N heartbeats per robot in a ring buffer.
    - Derives effective "reachable" based on last_seen vs threshold.
    - Tracks whether effective reachable changed (for event triggering).
    """

    def __init__(self, max_heartbeats: int = MAX_HEARTBEATS_PER_ROBOT, threshold_secs: float = HEARTBEAT_REACHABLE_THRESHOLD_SECS):
        self._robots: Dict[str, RobotHealthState] = {}
        self._host_port_to_robot_id: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._max_heartbeats = max_heartbeats
        self._threshold_secs = threshold_secs

    def record_heartbeat(
        self,
        robot_id: str,
        reachable: bool,
        busy: Optional[bool],
        ts: Optional[float],
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> bool:
        """
        Record a heartbeat for a robot.

        Args:
            robot_id: The robot's identifier.
            reachable: Whether the robot reported itself reachable.
            busy: Whether the robot is busy (optional).
            ts: Timestamp of the heartbeat; uses current time if None.
            host: Task server host (optional); used to index by host:port for fleet matching.
            port: Task server port (optional); used to index by host:port for fleet matching.

        Returns:
            True if effective "reachable" status changed (for event triggering), False otherwise.
        """
        now = ts if ts is not None else time.time()
        hb = Heartbeat(ts=now, reachable=reachable, busy=busy)

        with self._lock:
            state = self._robots.get(robot_id)
            if state is None:
                state = RobotHealthState()
                state.history = deque(maxlen=self._max_heartbeats)
                self._robots[robot_id] = state

            prev_effective = self._effective_reachable(state)

            state.history.append(hb)
            state.last_seen = now
            state.last_reachable = reachable
            if busy is not None:
                state.last_busy = busy

            if host is not None and port is not None:
                self._host_port_to_robot_id[f"{host}:{port}"] = robot_id

            new_effective = self._effective_reachable(state)
            return prev_effective != new_effective

    def get_health(self, robot_id: str) -> Optional[Dict]:
        """
        Get health summary for a single robot.

        Returns None if robot is unknown.
        """
        with self._lock:
            state = self._robots.get(robot_id)
            if state is None:
                return None
            return self._state_to_summary(robot_id, state)

    def get_all_health(self) -> Dict[str, Dict]:
        """
        Get health summary for all known robots.

        Returns a dict keyed by robot_id and also by "host:port" when provided in heartbeats,
        so the gateway can match registered robots (user-chosen name + host/port) by host:port.
        """
        with self._lock:
            out = {rid: self._state_to_summary(rid, st) for rid, st in self._robots.items()}
            for hp_key, rid in self._host_port_to_robot_id.items():
                if rid in self._robots:
                    out[hp_key] = self._state_to_summary(rid, self._robots[rid])
            return out

    def check_timeouts(self) -> List[str]:
        """
        Scan all robots and return IDs of those whose effective reachable
        just transitioned from True to False due to heartbeat timeout.

        Call this periodically from a background task so the service can
        push health_changed events even when no new heartbeats arrive.
        """
        timed_out: List[str] = []
        with self._lock:
            for rid, state in self._robots.items():
                was_reachable = state.last_reachable and state.last_seen != 0.0
                now_effective = self._effective_reachable(state)
                if was_reachable and not now_effective:
                    state.last_reachable = False
                    timed_out.append(rid)
        return timed_out

    def _effective_reachable(self, state: RobotHealthState) -> bool:
        """Derive effective reachable from last_seen and threshold."""
        if state.last_seen == 0.0:
            return False
        return (time.time() - state.last_seen) <= self._threshold_secs and state.last_reachable

    def _state_to_summary(self, robot_id: str, state: RobotHealthState) -> Dict:
        return {
            "robot_id": robot_id,
            "last_seen": state.last_seen,
            "reachable": self._effective_reachable(state),
            "busy": state.last_busy,
        }


# Module-level singleton instance
_store: Optional[HeartbeatStore] = None


def get_heartbeat_store() -> HeartbeatStore:
    """Get or create the singleton HeartbeatStore instance."""
    global _store
    if _store is None:
        _store = HeartbeatStore()
    return _store
