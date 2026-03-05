import pandas as pd
import matplotlib.pyplot as plt
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from version_index import VERSION_INDEX

FIGURE_NAME = f"../data/plots/training_summary_{VERSION_INDEX}.png"
DATA_FILE = f"../data/metrics/training_metrics_{VERSION_INDEX}.csv"

df = pd.read_csv(DATA_FILE)

y_columns = [
    "ep_rew_mean",
    "ep_len_mean",
    "value_loss",
    "approx_kl",
    "entropy_loss",
    "explained_variance",
    "avg_fruit",
    "avg_decayfruit",
    "avg_enemyfruit",
]

n_cols = 2
n_rows = math.ceil(len(y_columns) / n_cols)

plt.figure(figsize=(14, 5 * n_rows))

for i, col in enumerate(y_columns, start=1):
    plt.subplot(n_rows, n_cols, i)
    if col in df.columns:
        plt.plot(df["total_timesteps"], df[col])
        plt.title(col.replace("_", " ").title())
        plt.xlabel("Timesteps")
        plt.ylabel(col.replace("_", " ").title())
    else:
        print(f"Warning: column '{col}' not found in DataFrame.")

plt.tight_layout()
plt.savefig(FIGURE_NAME, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

# import pandas as pd
# import matplotlib.pyplot as plt

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from version_index import VERSION_INDEX

# FIGURE_NAME = f"../data/plots/training_summary_{VERSION_INDEX}.png"
# DATA_FILE = f"../data/metrics/training_metrics_{VERSION_INDEX}.csv"

# # load data
# df = pd.read_csv(DATA_FILE)

# # create figure
# plt.figure(figsize=(14, 10))

# # 1. Mean episode reward
# plt.subplot(3, 2, 1)
# plt.plot(df["total_timesteps"], df["ep_rew_mean"])
# plt.title("Mean Episode Reward")
# plt.xlabel("Timesteps")
# plt.ylabel("Reward")

# # 2. Mean episode length
# plt.subplot(3, 2, 2)
# plt.plot(df["total_timesteps"], df["ep_len_mean"])
# plt.title("Mean Episode Length")
# plt.xlabel("Timesteps")
# plt.ylabel("Length")

# # 3. Value loss
# plt.subplot(3, 2, 3)
# plt.plot(df["total_timesteps"], df["value_loss"])
# plt.title("Value Loss")
# plt.xlabel("Timesteps")
# plt.ylabel("Loss")

# # 4. Approx KL
# plt.subplot(3, 2, 4)
# plt.plot(df["total_timesteps"], df["approx_kl"])
# plt.title("Approx KL Divergence")
# plt.xlabel("Timesteps")
# plt.ylabel("KL")

# # 5. Entropy loss
# plt.subplot(3, 2, 5)
# plt.plot(df["total_timesteps"], df["entropy_loss"])
# plt.title("Entropy Loss")
# plt.xlabel("Timesteps")
# plt.ylabel("Entropy")

# # 6. Explained variance
# plt.subplot(3, 2, 6)
# plt.plot(df["total_timesteps"], df["explained_variance"])
# plt.title("Explained Variance")
# plt.xlabel("Timesteps")
# plt.ylabel("Variance")

# plt.tight_layout()

# # save
# plt.savefig(FIGURE_NAME, dpi=300, bbox_inches="tight")
# plt.show() 

# plt.close()