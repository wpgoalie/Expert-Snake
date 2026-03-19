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
fruit_pattern = re.compile(r"\{([^}]*)\}")

with open(LOG_FILE, "r") as f:
    for line in f:
        match = line_pattern.search(line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()

            try:
                value = float(value) if "." in value else int(value)
            except ValueError:
                pass

            current[key] = value

        if "FruitCounts" in line:
            fruit_match = fruit_pattern.search(line)
        
            if fruit_match and rows:
                fruit_items = fruit_match.group(1).split(",")
        
                for item in fruit_items:
                    name, val = item.split(":")
                    name = name.strip().strip("'").lower()
                    val = int(val.strip())
        
                    rows[-1][f"avg_{name}"] = val

        if line.startswith("-") and current:
            if "total_timesteps" in current:
                rows.append(current)
                current = {}

df = pd.DataFrame(rows)

if "total_timesteps" in df.columns:
    df = df.sort_values("total_timesteps")

df.to_csv(SAVE_PATH, index=False)
print(df.head())