#!/usr/bin/env python3
"""Launcher script for the Sikap-Salita live demo."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("live_demo.app:app", host="0.0.0.0", port=8000)
