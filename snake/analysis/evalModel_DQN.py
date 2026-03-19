import numpy as np
import pandas as pd
from stable_baselines3 import PPO, DQN
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from version_index import VERSION_INDEX
from rlEnvironment_DQN import snakeRLEnvironment

# parameters
MODEL_TYPE = "dqn"  # "ppo" or "dqn"
MODEL_PATH = f"../data/models/{MODEL_TYPE}_snake_{VERSION_INDEX}.zip"

NUM_EPISODES = 50
VIDEO_FOLDER = "../data/videos/"
SAVE_CSV = f"../data/metrics/eval_metrics_{VERSION_INDEX}.csv"
SUMMARY_CSV = f"../data/metrics/summary_eval_metrics_{VERSION_INDEX}.csv"
RECORD_VIDEO = True


# environment
env = snakeRLEnvironment(render_mode="rgb_array")
env = RecordEpisodeStatistics(env, buffer_length=NUM_EPISODES)

# video recording
if RECORD_VIDEO:
    env = RecordVideo(
        env,
        video_folder=VIDEO_FOLDER,
        name_prefix="eval",
        episode_trigger=lambda x: x % 5 == 0
    )

# load model
if MODEL_TYPE == "ppo":
    model = PPO.load(MODEL_PATH)
elif MODEL_TYPE == "dqn":
    model = DQN.load(MODEL_PATH)
else:
    raise ValueError("MODEL_TYPE must be 'ppo' or 'dqn'")


# evaluation
episode_rewards = []
episode_lengths = []
episode_scores = []
episode_fruit_cnts = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    ep_length = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        ep_reward += reward
        ep_length += 1

    episode_scores.append(env.unwrapped.score())
    episode_fruit_cnts.append(env.unwrapped.fruit_stats())
    episode_rewards.append(ep_reward)
    episode_lengths.append(ep_length)


# save per-episode metrics
metrics_data = {
    "episode": np.arange(1, NUM_EPISODES + 1),
    "reward": episode_rewards,
    "length": episode_lengths,
    "score": episode_scores,
}

fruit_keys = list(episode_fruit_cnts[0].keys())
for key in fruit_keys:
    metrics_data[key] = [fc[key] for fc in episode_fruit_cnts]

metrics_df = pd.DataFrame(metrics_data)
metrics_df["avg_reward_per_step"] = metrics_df["reward"] / metrics_df["length"]


# summary statistics
summary_data = {
    "average_reward": np.mean(episode_rewards),
    "std_reward": np.std(episode_rewards),
    "average_length": np.mean(episode_lengths),
    "average_score": np.mean(episode_scores),
    "std_length": np.std(episode_lengths),
    "max_reward": np.max(episode_rewards),
    "min_reward": np.min(episode_rewards),
}

for key in fruit_keys:
    summary_data[f"avg_{key.lower()}"] = np.mean(metrics_df[key])

summary_df = pd.DataFrame([summary_data])


# save csv
metrics_df.to_csv(SAVE_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)

print(f"\nIndividual episode metrics saved to {SAVE_CSV}")
if RECORD_VIDEO:
    print(f"Video saved to folder: {VIDEO_FOLDER}")

env.close()