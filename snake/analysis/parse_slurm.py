import re
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from version_index import VERSION_INDEX

LOG_FILE = f"../data/keep_slurm/slurm-{VERSION_INDEX}.out"
SAVE_PATH = f"../data/metrics/training_metrics_{VERSION_INDEX}.csv"

rows = []
current = {}

line_pattern = re.compile(r"\|\s+([^|]+?)\s+\|\s+([^|]+?)\s+\|")

with open(LOG_FILE, "r") as f:
    for line in f:
        match = line_pattern.search(line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()

            # Try numeric conversion
            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass

            current[key] = value

        # Separator = end of block
        if line.startswith("-") and current:
            if "iterations" in current:
                rows.append(current)
            current = {}

df = pd.DataFrame(rows)

df = df.sort_values("iterations")

df.to_csv(SAVE_PATH, index=False)
print(df.head())