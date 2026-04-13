#!/usr/bin/env python3
"""Launcher wrapper for ByteTrack that applies NumPy compatibility shim."""

import sys
from pathlib import Path

# Apply numpy shim before any other imports
from src.modules.trackers import numpy_shim  # noqa: F401

# Get ByteTrack root from environment or default
bytetrack_root = Path(__file__).resolve().parents[4] / "origin" / "ByteTrack"
if not bytetrack_root.exists():
    # Try alternative location
    bytetrack_root = Path("/home/fzliang/origin/ByteTrack")

# Add ByteTrack to path
sys.path.insert(0, str(bytetrack_root))

# Import and run ByteTrack's demo_track
from tools.demo_track import main

if __name__ == "__main__":
    main()
