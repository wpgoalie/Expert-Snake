import gymnasium as gym
import numpy as np
# import spinup
from rlEnvironment import snakeRLEnvironment
# from gymnasium.wrappers import StepAPICompatibility
# from spinup.algos.pytorch.ppo.core import MLPActorCritic

def environment_function():
    snakeEnvironment = snakeRLEnvironment()
    return StepAPICompatibility(snakeEnvironment)
    
# def main():        
#     spinup.ppo_pytorch(environment_function, actor_critic=MLPActorCritic, ac_kwargs={}, seed=0, steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=0.0003, vf_lr=0.001, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs={}, save_freq=10)

if __name__ == '__main__':
    set_environment = snakeRLEnvironment(6, 6)
    obs = set_environment.reset()
    
    for i in range(50):
        current_action = set_environment.action_space.sample()
        print("CURRENT ACTION", current_action)
        observation, reward, terminated, truncated, info = set_environment.step(current_action)
        print("REWARD", reward)

        if terminated:
            print("RESET")
            obs = set_environment.reset()
    # main()
        