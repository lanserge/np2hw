# Copyright 2026 Serge Rabyking
# SPDX-License-Identifier: MIT
"""The toy library beside these tests is found as any library is: by the
import path."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
