import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from rlEnvironment import snakeRLEnvironment

# parameters
MODEL_PATH = "../data/models/ppo_snake.zip"
NUM_EPISODES = 50
VIDEO_FOLDER = "../data/videos/"
SAVE_CSV = "../data/misc/eval_metrics.csv"
SUMMARY_CSV = "../data/misc/summary_eval_metrics.csv"
RECORD_VIDEO = True

env = snakeRLEnvironment(render_mode="rgb_array")
env = RecordEpisodeStatistics(env, buffer_length=NUM_EPISODES)

# set up recording
if RECORD_VIDEO:
    env = RecordVideo(
        env,
        video_folder=VIDEO_FOLDER,
        name_prefix="eval",
        episode_trigger=lambda x: x % 5 == 0
    )

# load model
model = PPO.load(MODEL_PATH)

# evaluation
episode_rewards = []
episode_lengths = []
episode_scores = []

for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    ep_length = 0
    score = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        ep_reward += reward
        ep_length += 1

    episode_scores.append(env.unwrapped.score())
    episode_rewards.append(ep_reward)
    episode_lengths.append(ep_length)

# save metrics
metrics_df = pd.DataFrame({
    "episode": np.arange(1, NUM_EPISODES + 1),
    "reward": episode_rewards,
    "length": episode_lengths,
    "score": episode_scores,
})
metrics_df["avg_reward_per_step"] = metrics_df["reward"] / metrics_df["length"]
summary_df = pd.DataFrame([{
    "average_reward": np.mean(episode_rewards),
    "std_reward": np.std(episode_rewards),
    "average_length": np.mean(episode_lengths),
    "average_score" : np.mean(episode_scores),
    "std_length": np.std(episode_lengths),
    "max_reward": np.max(episode_rewards),
    "min_reward": np.min(episode_rewards),
}])

metrics_df.to_csv(SAVE_CSV, index=False)
summary_df.to_csv(SUMMARY_CSV, index=False)

print(f"\nIndividual episode metrics saved to {SAVE_CSV}")
if RECORD_VIDEO:
    print(f"Video saved to folder: {VIDEO_FOLDER}")

env.close()