import numpy as np
# from pathlib import Path
import os 
import glob

# import gymnasium as gym
from rlEnvironment import snakeRLEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from version_index import VERSION_INDEX

CHECKPOINT_DIR = "./data/model_checkpoints" # checkpoints saved into data so that cancelled jobs can be continued
MODEL_NAME = f"ppo_snake_{VERSION_INDEX}"
FINAL_MODEL = f"{MODEL_NAME}.zip" # final model saved to snake/

def environment_function():
    return snakeRLEnvironment()
    
def main(): 
    # if a fully trained model exists (based on VERSION_INDEX) in snake/, stop program
    if os.path.exists(FINAL_MODEL):
        print("A completely trained model already exists.")
        print("Please clear or move relevant files before training a new model.")
        exit()
        
    # Add video recording for every episode
    env = RecordVideo(
        snakeRLEnvironment(render_mode = "rgb_array"),
        video_folder="snake-agent",    # Folder to save videos
        name_prefix="eval",               # Prefix for video filenames
        episode_trigger=lambda x: True    # Record every episode
        # episode_trigger=lambda x: x % 100 == 0 # record every 100 episodes
    )

    env = RecordEpisodeStatistics(env, buffer_length = 15000)

    # make directory if it doesn't exist
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    # load checkpoint files
    checkpoint_files = glob.glob(f"{CHECKPOINT_DIR}/{MODEL_NAME}_*_steps.zip")

    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env)
    else:
        print("No checkpoints found. Starting fresh training.")
        model = PPO("MultiInputPolicy", env, verbose=1)

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,          # save every 500k steps
        save_path=CHECKPOINT_DIR,
        name_prefix=MODEL_NAME
    )
    
    # model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=2_500_000, callback=checkpoint_callback, reset_num_timesteps=False)
    model.save(FINAL_MODEL)

    avg_reward = np.average(env.return_queue)
    avg_length = np.average(env.length_queue)

    env.close()

    print(f'Average Reward: {avg_reward:.2f}')
    print(f'Average Episode Length: {avg_length:.1f}')

if __name__ == '__main__':
    main()
        