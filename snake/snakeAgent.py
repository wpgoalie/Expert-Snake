import gymnasium as gym
import numpy as np
from rlEnvironment import snakeRLEnvironment
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from pathlib import Path

from version_index import VERSION_INDEX

def environment_function():
    return snakeRLEnvironment()
    
def main():
    # Add video recording for every episode
    env = RecordVideo(
        snakeRLEnvironment(render_mode = "rgb_array"),
        video_folder="snake-agent",    # Folder to save videos
        name_prefix="eval",               # Prefix for video filenames
        episode_trigger=lambda x: True    # Record every episode
    )

    env = RecordEpisodeStatistics(env, buffer_length = 15000)
    
    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=5_000_000)
    model.save(f"ppo_snake_{VERSION_INDEX}")
    env.close()

    avg_reward = np.average(env.return_queue)
    avg_length = np.average(env.length_queue)

    print(f'Average Reward: {avg_reward:.2f}')
    print(f'Average Episode Length: {avg_length:.1f}')

if __name__ == '__main__':
    main()
        