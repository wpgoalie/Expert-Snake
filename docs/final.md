---
layout: default
title: Final Report
---

## Project Summary:

Our RL project focuses on the Snake game but a specific variation of it, the "cheese" variation. In this variation, the snake agent has the ability to maneuver through its own body, where every other body tile starting from the tile following the head of the snake is non-collidable. The other body tiles are still collidable, meaning that the snake has to time its turns to be able to turn into itself and stay alive. This addition to the original game becomes very useful whenever the snake agent gets cornered into a corner of the boundaries. At every point in the game, the snake agent has the choice to turn left or right relative to its current position, but it could also continue forward without any turn as well. Whenever the snake collects an apple, this increases the length of the snake agent after a small time buffer to give time for the snake agent to adapt to the new snake length. Our project uses the PPO (Proximal Policy Optimization) algorithm to train our snake agents to maximize the total game score and to efficiently choose turns while doing so. Our project focused on a grid of size 30 by 24 tiles, with an original snake body start of 4 segments.

This project consists of two phases:

### Phase 1: One Fruit Type

In this phase, there was only one fruit, the regular fruit, which increases the score by 1 when eaten. In this phase, the snake agent only had observations to the snake head's position and the fruits position. We later added another observation, which consisted of letting the snake agent know which next moves resulted in danger or not. The main goals of this phase were to try to maximize fruit score and to inspect and eliminate undesirable behavior so that these issues don't persist in phase 2.

<img height="300" alt="phase_1_visual" src="/images/phase_1_visual.jpg" />

### Phase 2: Three Fruit Types

In this phase, two additional fruit were added: the enemy fruit and the decay fruit. The enemy fruit followed a set square path and would decrease the score by 1 whenever either the snake ate this fruit or the fruit itself ran into the snake's body. The Decay fruit started with an initial score increase reward of 5 when eaten, but this reward decreases by 1 every 5 steps in the training process, where this fruit respawns at a new location when it hits a score increase reward of 0. On top of the observation space of phase 1, we added an enemy fruit danger observation, which tells the snake agent if the next move of the enemy fruit results in a collision with the snake itself. Also, instead of just observing the location of one fruit, there are now three fruit locations in the observation space. The main goals of this phase were to maximize the fruit score by eating both regular fruit and decay fruit efficiently while also avoiding enemy fruit.

<img height="300" alt="phase_1_visual" src="/images/phase_2_visual.jpg" />

## Approach

## Evaluation

## Resources Used

- [Gymnasium Custom Environment Documentation](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) used for creating a custom Snake game RL Environment
- [Gymnasium Video Recording Wrapper Documentation](https://gymnasium.farama.org/main/_modules/gymnasium/wrappers/record_video/) used for visualizing the training process of the snake agent
- [Stable-Baselines3 Callbacks Documentation](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html) used for tracking score statistics and storing model checkpoints
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html) used for displaying mean episode rewards, mean episode lengths, value/entropy losses, explained variance values, and approximate KL divergence values
- [Stable-Baselines3 PPO Documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) used for implementation on a PPO algorithm setup
- [OpenAI Spinning Up Reinforcement Learning Introductory Overview](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) used to understand the role of observation spaces and how they work with chosen actions during the training process
- [Article on Reinforcement Learning in Snake](https://xiaoyang-rebecca.github.io/posts/2025/01/rl-snake/) used to understand the typical environment setup used to train an agent on the Snake game
- [GeeksforGeeks Snake Pygame Implementation Tutorial](https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/) used for setting up the base code of the classic Snake game using the Pygame library
- [Pygame Documentation](https://www.pygame.org/docs/) used to add on additional features to the base code of the Snake game, including the "snake" variation
- [ChatGPT](https://chatgpt.com/) used for understanding slurm outputs, debugging the reward system and observation space, and writing code to parse slurm files