#!/usr/bin/env python
"""Entry point for video search agent."""
import uvicorn
from backend.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
