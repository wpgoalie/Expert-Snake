import numpy as np
import os 
import glob

from rlEnvironment_DQN import snakeRLEnvironment
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

from version_index import VERSION_INDEX

CHECKPOINT_DIR = "./data/model_checkpoints" # checkpoints saved into data so that cancelled jobs can be continued
MODEL_NAME = f"dqn_snake_{VERSION_INDEX}"
FINAL_MODEL = f"{MODEL_NAME}.zip" # final model saved to snake/


class FruitLoggingCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.n_calls % 10000 == 0:
            env = self.training_env.envs[0].unwrapped
            fruit_counts = env.fruit_stats()
            print(f"FruitCounts | {fruit_counts} |")
        return True


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
        model = DQN.load(latest_checkpoint, env=env)
    else:
        print("No checkpoints found. Starting fresh training.")
        # model = DQN(
        #     "MlpPolicy",
        #     env,
        #     learning_rate=1e-4,
        #     buffer_size=100000,
        #     learning_starts=10000,
        #     batch_size=64,
        #     gamma=0.99,
        #     train_freq=4,
        #     target_update_interval=1000,
        #     exploration_fraction=0.4,
        #     exploration_final_eps=0.1,
        #     verbose=1
        # )
    # model 4
        # model = DQN(
        #     "MlpPolicy",
        #     env,
        #     learning_rate=5e-4,
        #     buffer_size=200000,
        #     learning_starts=2000,
        #     batch_size=128,
        #     gamma=0.99,
        #     train_freq=4,
        #     target_update_interval=1000,
        #     exploration_fraction=0.1,
        #     exploration_final_eps=0.02,
        #     policy_kwargs=dict(
        #         net_arch=[256, 256],
        #     ),
        #     verbose=1
        # )
    # model 5
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            buffer_size=200000,
            learning_starts=20000,
            batch_size=128,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            exploration_fraction=0.4,
            exploration_final_eps=0.05,
            verbose=1
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=500_000,          # save every 500k steps
        save_path=CHECKPOINT_DIR,
        name_prefix=MODEL_NAME
    )
    fruit_callback = FruitLoggingCallback()
    callback_list = CallbackList([checkpoint_callback, fruit_callback])
    
    model.learn(total_timesteps=1_000_000, callback=callback_list, reset_num_timesteps=False)
    model.save(FINAL_MODEL)

    avg_reward = np.average(env.return_queue)
    avg_length = np.average(env.length_queue)

    env.close()

    print(f'Average Reward: {avg_reward:.2f}')
    print(f'Average Episode Length: {avg_length:.1f}')


if __name__ == '__main__':
    main()
        