---
layout: default
title: Final Report
---

## Project Summary:

Our RL project focuses on the Snake game but a specific variation of it, the "cheese" variation. In this variation, the snake agent has the ability to maneuver through its own body, where every other body tile starting from the tile following the head of the snake is non-collidable. The other body tiles are still collidable, meaning that the snake has to time its turns to be able to turn into itself and stay alive. This addition to the original game becomes very useful whenever the snake agent gets cornered into a corner of the boundaries. At every point in the game, the snake agent has the choice to turn left or right relative to its current position, but it could also continue forward without any turn as well. Whenever the snake collects an apple, this increases the length of the snake agent after a small time buffer to give time for the snake agent to adapt to the new snake length. Our project uses the PPO (Proximal Policy Optimization) algorithm to train our snake agents to maximize the total game score and to efficiently choose turns while doing so. Our project focused on a grid of size 30 by 24 tiles, with an original snake body start of 4 segments.

This project consists of two phases:

### Phase 1: One Fruit Type

In this phase, there was only one fruit, the regular fruit, which increases the score by 1 when eaten. In this phase, the snake agent only had observations to the snake head's position and the fruits position. We later added another observation, which consisted of letting the snake agent know which next moves resulted in danger or not. The main goals of this phase were to try to maximize fruit score and to inspect and eliminate undesirable behavior so that these issues don't persist in phase 2.

![Phase 1 Visual](/images/phase_1_visual.jpg)

### Phase 2: Three Fruit Types

In this phase, two additional fruit were added: the enemy fruit and the decay fruit. The enemy fruit followed a set square path and would decrease the score by 1 whenever either the snake ate this fruit or the fruit itself ran into the snake's body. The Decay fruit started with an initial score increase reward of 5 when eaten, but this reward decreases by 1 every 5 steps in the training process, where this fruit respawns at a new location when it hits a score increase reward of 0. On top of the observation space of phase 1, we added an enemy fruit danger observation, which tells the snake agent if the next move of the enemy fruit results in a collision with the snake itself. Also, instead of just observing the location of one fruit, there are now three fruit locations in the observation space. The main goals of this phase were to maximize the fruit score by eating both regular fruit and decay fruit efficiently while also avoiding enemy fruit.

![Phase 2 Visual](/images/phase_2_visual.jpg)

## Approach

We focused on using the PPO (Proximal Policy Optimization) algorithm to train our agents in both phases of our project. We decided on using this algorithm as our action space is discrete and PPO works well with discrete action spaces, where the snake could either turn left, right, or continue going forward at each point in the game. We decided to train on 1 million timesteps for both phases, where during the initial period of phase 1 we trained on 25k-50k timesteps in order to test and see how our setup was doing. Specifically, we use the clip version of the PPO algorithm. The clip version specifically prevents drastic updates to the policy itself, where drastic updates could lead to forgetting good behavior. Clipping the probability ratio prevents the snake agent from making drastic turns, reducing the risk of the snake itself running into a terminated state (running into a wall, dying to an enemy fruit, running into itself). Policies are updated using:

$$
{E}_{(s,a)∼p\overline{\theta}}[L\frac{\theta}{\theta}(s,a)] 
$$

where L is given by:

$$
L\frac{\theta}{\theta}(s,a)=min(\rho\frac{\theta}{\theta}(a|s)A_{\overline{\theta}}(s,a), {A_{\overline{\theta}}(s,a)}+{|\epsilon A_{\overline{\theta}}(s,a)|}
$$

For the environment setup, we adapted a classic snake game pygame provided by [github](https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/) into the cheese variation with configurable parameters such as board size. We then linked this pygame with a Gymnasium environment.

### Phase 1 Observation Space

Our observation space for phase 1 consists of the following:
- "agent": snake head's current position on the grid (2D coordinate)
- "target": current position of the fruit
- "danger": indicates whether moving in each cardinal direction (up, down, left, right) would result in collision with itself or a wall, using a dictionary of where dangers are based on direction

Initially, the agent only observed its head and the fruit. This led to poor learning and the agent frequently ran into its own body. We noticed that the agent wasn’t taking into account its body, resulting in the snake agent constantly running into itself during training, so we added an extra "danger" observation that would help the snake detect the board boundaries or parts of its tail based on the next move.

### Phase 2 Observation Space

Our observation space for phase 2 consists of the following:
- "agent": snake head's current position on the grid (2D coordinate)
- "fruits": current positions and normalized values of all three fruits (regular fruit, decay fruit, enemy fruit) in the form: `[x, y, normalized_value]` for each fruit stored in a 2D array
- "danger": indicates whether moving in each cardinal direction (up, down, left, right) would result in collision with itself or a wall, using a dictionary of where dangers are based on direction
- "enemy_danger": indicates whether the next move of the enemy fruit on its set square path would result in either the snake eating the fruit or a collision with the snake's body, using a dictionary of where dangers are based on direction

### Phase 1 Reward System

For our phase 1 reward system, our original reward system was:
- **+1** if the score increased after the chosen action
- **-1** if the game terminated (snake agent ran into a wall or itself)
- **-0.01** for every action the snake took

<figure>
<video width="320" height="240" controls>
  <source src="./images/eval-episode-64.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>Example of a snake rushing towards the wall instead of the fruit in order to end the run.</figcaption>
</figure>

However, with this reward system, the snake agent never initially got to the apples, so it would always try to run into a wall to end its episode earlier, resulting in a less negative reward total than if they had continued exploring. It also circled apples if it ever got to them and took many unnecessary turns This is why on top of these rewards, we added another reward mechanic, which was based on the Euclidean distance between the snake head and the fruit:
- **+0.1** if the chosen action resulted in getting closer to the fruit
- **-0.1** if the chosen action resulted in getting farther from the fruit

Our final phase 1 reward system was:
- **+2** if the score increased after the chosen action
- **-1** if the snake agent ran into itself
- **-1.5** if the snake agent ran into a wall
- **-0.01** for every action the snake took
- **+0.1** if the chosen action resulted in the snake agent getting closer to the fruit
- **-0.1** if the chosen action resulted in the snake agent getting farther from the fruit

<figure>
<video width="320" height="240" controls>
  <source src="./images/eval-episode-64.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>Example of a run with the final reward system for phase 1.</figcaption>
</figure>

This updated reward system helps the agent know that the fruit is the main goal of the training process, resulting in the snake agent focusing its vision on the fruit rather than finding a fast way to stop its depletion of the reward in its current episode run. This new reward system also resulted in the snake agent using the “cheese” mechanic to its advantage when it was cornered and reduced the number of turns the snake agent took.

### Phase 2 Reward System

For our phase 2 reward system, our original reward system was the same as phase 1's final reward system.

<figure>
<video width="320" height="240" controls>
  <source src="./images/phase2_initial.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>Example of a snake constantly running into the enemy fruit.</figcaption>
</figure>

As this second phase added two new fruits (the decay fruit and the enemy fruit) to the Snake game, we observed new issues:
- The snake agent would constantly run into the enemy fruit
- The snake agent couldn't decide whether to eat the regular fruit or the decay fruit, going back and forth without eating either of them
- The snake agent ignored the regular fruit in many cases, chasing solely the decay fruit, even if the regular fruit was the better choice

In order to fix these issues, we decided to give feedback on the progress (distance) towards the nearest fruit in order to encourage efficient fruit eating to maximize the score of the game. We normalized distances based on the current score and maximum score of each fruit, as unlike phase 1, there are now additional fruits with score rewards other than +1. Adding onto this, penalties were given after each action if the action resulted in the closest fruit to the snake agent being the enemy fruit. To further encourage the snake agent to avoid running into the enemy fruit, a penalty was given on score decrease as well. Normalization is done by the formula:

$$
S=0.5\times\frac{d_{prev} - d_{curr}}{\sqrt{L^2_x+L^2_y}}
$$

Our final phase 2 reward system was:
- **+20** * (new score - previous score) on score increase
- **−15** * (previous  score - new score) on score decrease
- **-12** if the game terminated (snake agent ran into a wall, itself, or died to an enemy fruit)
- **-0.01** for surviving
- **-0.003** for changing directions
- **-0.05** for moving away from any fruit
- **+0.5** * (normalized distance change to closest fruit) for feedback on snake progress towards fruits
- **+S** move toward regular fruit
- **2S** move toward enemy fruit
- **+(1 + 2 * value_ratio) * S** for eating decay fruit

<figure>
<video width="320" height="240" controls>
  <source src="./images/phase2_final.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>Example of a run with the final reward system for phase 2.</figcaption>
</figure>

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
- [Google Snake Game Cheese Mode Wiki Article](https://google-snake.fandom.com/wiki/Cheese_Mode) used for learning more about the "cheese" variation of the Snake game
- [ChatGPT](https://chatgpt.com/) used for understanding slurm outputs, debugging the reward system and observation space, and writing code to parse slurm files