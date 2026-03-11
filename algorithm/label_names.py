# -*- coding: utf-8 -*-
"""
Label name mapping — extracted from datasets/RPS/label_name.py
Keep this file as the single source of truth for class labels.
"""

# Maps English model label → display name
# Extend / replace with your actual dataset labels.
CHINESE_NAME: dict[str, str] = {
    "rock": "石头",
    "paper": "布",
    "scissors": "剪刀",
}

# Ordered list used for count tables and chart axes
LABEL_LIST: list[str] = list(CHINESE_NAME.values())
