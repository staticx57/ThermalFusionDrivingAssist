#!/bin/bash
# Run VPI-accelerated thermal road monitor with proper EGL configuration
#
# Usage:
#   ./run_vpi.sh                  # Default 2x scaling
#   ./run_vpi.sh --scale 2.5      # Custom scaling
#   ./run_vpi.sh --fullscreen     # Fullscreen mode

export EGL_PLATFORM=surfaceless
python3 main_vpi.py "$@"