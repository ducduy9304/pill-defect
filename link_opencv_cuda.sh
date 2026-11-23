#!/usr/bin/env bash
# Link the locally built CUDA OpenCV into .venv. Re-run whenever .venv is recreated.
set -euo pipefail
cd "$(dirname "$0")"

echo ~/.local/opencv/5.x-cuda13.3/lib/python3.14/site-packages > .venv/lib/python3.14/site-packages/cv2_cuda.pth
.venv/bin/python -c "import cv2; print('OpenCV', cv2.__version__, '| CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())"
