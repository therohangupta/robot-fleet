#!/usr/bin/env python3
"""
CLI entrypoint for the Telemetry service.

Usage: python -m services.telemetry.src [--port PORT]
"""

import argparse
import uvicorn

from .config import TELEMETRY_PORT


def main():
    parser = argparse.ArgumentParser(description="Start the Telemetry service")
    parser.add_argument("--port", type=int, default=TELEMETRY_PORT, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(
        "services.telemetry.src.main:app",
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
