"""
Telemetry service entrypoint.

Run with: uvicorn services.telemetry.src.main:app --port 9000
Or: python -m services.telemetry.src (if __main__.py is present)
"""

from .app import app

# Re-export for uvicorn
__all__ = ["app"]
