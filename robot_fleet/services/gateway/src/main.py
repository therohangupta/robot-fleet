"""
Robot Fleet Dashboard - Entry Point

This module serves as the entry point for running the dashboard backend.
The actual application is defined in app.py.

Usage:
    uvicorn dashboard.backend.main:app --reload
    
Or from the project root:
    cd dashboard/backend && uvicorn main:app --reload
"""

from .app import app

# Re-export app for uvicorn
__all__ = ["app"]
