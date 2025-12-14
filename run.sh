#!/bin/bash
# SAM3 Object Detection Tool - Runner Script
# Usage: ./run.sh --input <images_dir> --classes <classes.txt> [options]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"

python "$SCRIPT_DIR/detect.py" "$@"
