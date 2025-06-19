#!/usr/bin/env python3
import asyncio
import argparse
import os
import signal
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ..server.service import serve

async def main():
    parser = argparse.ArgumentParser(description='Start the Fleet Manager server')
    parser.add_argument('--port', type=int, default=50051, help='Port to listen on')
    parser.add_argument('--reset-db', action='store_true', help='Reset database on startup')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose application logging')
    parser.add_argument('--sql-debug', action='store_true', help='Enable SQL debug logging')
    args = parser.parse_args()

    db_url = os.environ.get('DATABASE_URL')

    stop_event = asyncio.Event()

    def handle_signal(*_):
        print("\nReceived exit signal, shutting down gracefully...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            # add_signal_handler may not be implemented on Windows
            pass

    server_task = asyncio.create_task(
        serve(port=args.port, db_url=db_url, reset_db=args.reset_db, verbose=args.verbose, sql_debug=args.sql_debug)
    )
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nGraceful shutdown initiated.")
    finally:
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            pass

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user.")
        sys.exit(0)