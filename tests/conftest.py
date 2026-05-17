# Copyright (c) 2026 Scenema AI
# https://scenema.ai
# SPDX-License-Identifier: MIT

import sys
import os

# Add audio_core directly to path (avoids triggering root __init__.py)
pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
audio_core_path = os.path.join(pkg_root, "audio_core")
if audio_core_path not in sys.path:
    sys.path.insert(0, audio_core_path)
