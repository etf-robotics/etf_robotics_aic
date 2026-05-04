# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

# Register Gym environments.
from .tasks import *

# Register UI extensions when running inside Isaac Sim's UI extension context.
try:
    from .extension import *
except ModuleNotFoundError as exc:
    if exc.name is None or not exc.name.startswith("omni"):
        raise
