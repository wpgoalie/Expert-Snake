---
layout: default
title: Final Report
---

## Project Summary:

Our RL project focuses on the Snake game's "cheese" variation, where the snake nowhas a gap in between every single body part following the head, allowing it to maneuver through its own body. This addition to the original game introduces a lot more freedom of movement into the game, since now the snake does not have to actively work around its own body, but also has to be careful to time its manuevers correctly. Much like the classic version, the snake agent can choose to turn right, turn left, or continue going forward. The snake will also terminate once it collides with the wall or itself. Whenever the snake eats an apple, the snake will add a new body tile and gap. Our project uses the Proximal Policy Optimization (PPO) algorithm to train our snake agents to maximize the total game score and to efficiently navigate the board while doing so. Our project focused on a grid of size 30 by 24 tiles, with an the snake initially having 4 body tiles.

This project consists of two phases:

### Phase 1: Classic Cheese Variation

In this phase, we kept the basic cheese variation set up, as described above. The main goals of this phase were to try to maximize fruit score and to eliminate highly undesirable behavior early on so that these issues don't persist in phase 2, when our set up becomes much more complicated.

![Phase 1 Visual](/images/phase_1_visual.jpg)

### Phase 2: Additional Special Fruit

In this phase, two additional fruits were added: the decay fruit and the enemy fruit. The decay fruit (yellow) starts with an initial score of 5 and decreases by 1 every 5 steps, respawning at a new location when it hits 0. The enemy fruit is depicted as purple and follows a set square path. When it collides with the snake head or body, it decreases the score by 1 and also subtracts the tail body segment, meaning that the snake can also die if it eats too many enemy fruits.

Altogether, these features create a dynamic environment where the agent must plan ahead, balance immediate and future rewards, and adapt to changing conditions. A non-RL approach, such as a fixed algorithm, would struggle to handle the many complex, case-by-case situations that can arise. Even in the classic cheese variation, fully exploiting the gaps in the snake’s body is challenging, as the snake has increased freedom to move through itself in order to avoid death or efficiently collect fruit. With the addition of decaying and enemy fruit, the agent must also strategically choose which fruit to pursue based on location and timing. Machine learning algorithms are well suited to this type of dynamic environment, as they allow the agent to learn strategies from experience and make decisions that balance both short-term rewards and long-term outcomes. Through this project, we aim to discover what an optimal strategy looks like for our environment and understand how our RL agent can effectively navigate the snake’s gaps, weigh trade-offs to prioritize fruit, and adapt to a constantly changing game state in order to maximize its score.

![Phase 2 Visual](/images/phase_2_visual.jpg)

## Approach

We focused on using the Proximal Policy Optimization (PPO) algorithm to train our agents in both phases of our project. We decided on using this algorithm because our action space is discrete (at each time step, we have the option of left, right, or forward), which PPO works well with. We train the PPO agent with the `MultiInputPolicy` from `Stable-Baselines3`, since its support for dictionary observation spaces is beneficial for including multiple types of information in our observation space, as will be discussed in more detail later in the report. Specifically, we use the clip version of the PPO algorithm. This was since the clip version prevents drastic updates to the policy itself, allowing the agent can improve gradually through fine-tuning how it should navigate body gaps, prioritize fruits, and avoid hazards without ignoring previously effective strategies. Clipping the probability ratio prevents the agent from making drastic jumps that abandon successful behavior in place of erratic turns, which could easily land the snake in a terminated state since one wrong turn can cause it to overconsume enemy fruit and hit the wall or itself. Policies are updated using:

$$
{E}_{(s,a)∼p\overline{\theta}}[L\frac{\theta}{\theta}(s,a)] 
$$

where L is given by:

$$
L\frac{\theta}{\theta}(s,a)=min(\rho\frac{\theta}{\theta}(a|s)A_{\overline{\theta}}(s,a), {A_{\overline{\theta}}(s,a)}+{|\epsilon A_{\overline{\theta}}(s,a)|}
$$

For the environment setup, we adapted a classic snake game pygame provided by [github](https://www.geeksforgeeks.org/python/snake-game-in-python-using-pygame-module/) into the cheese variation and added configurable parameters such as board size and optional rendering and debug modes. We then linked this pygame with a Gymnasium environment. In phase 2, we made further adjustments to the snake game so that it can fully support all the different types of fruit and the behavior it induces on the snake.

We decided to train on 1 million timesteps for both phases. However, at the very beginning of phase 1, we did train on 25k-50k timesteps in order to verify our setup. Additionally, we kept the default PPO parameters because PPO is very sensitive to hyperparameters and can quickly become unstable, and whenever we made adjustments to its parameters the results were always significantly worse. These hyperparameters are defined in [Stable-Baselines3 PPO documentation](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), and are as follows:

[insert hyperparam table here]

### Phase 1
#### Observation Space

Our observation space for phase 1 consists of the following:
- "agent": snake head's current position on the grid (2D coordinate)
- "target": current position of the fruit
- "danger": indicates whether moving in each cardinal direction (up, down, left, right) would result in an immediate collision with itself or a wall, using a dictionary that maps direction with the danger (or lack of)

Initially, the agent only observed its head and fruit location. This led to poor learning as the agent completely ignored its own body, and frequently ran into it as a result. We added an extra "danger" observation that would help the snake detect the board boundaries or parts of its tail based on the next move.

#### Reward System

For our phase 1 reward system, we started with an extremely simple reward system:
- **+1** if the score increased after the chosen action
- **-1** if the game terminated (snake agent ran into a wall or itself)
- **-0.01** for every action the snake took to encourage fruit consumption

<figure>
<video width="320" height="240" controls>
  <source src="./images/eval-episode-64.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>Example of a snake rushing towards the wall instead of the fruit in order to end the run.</figcaption>
</figure>

As we tuned our reward system, we observed several persistent undesirable behaviors such as circling the fruit instead of eating it, intentionally killing itself early on, and taking excessively long and winding paths. Circling the fruit was due to us rewarding the snake based on distance, and so it began prioritizing staying close instead of eating the fruit and hunting for the next one. Intentionally dying early on was because we introduced a survival pressure hoping that it would motivate the snake into finding the fruit faster, but because our initial survival pressure was too high the snake decided that it could get a less negative reward by dying soon. Excessive turning was also due to its desire to shorten the distance quickly at each time step, and so we penalized turning by a little. We also discovered that we had to be very careful with our penalizations, since being even a little too aggressive would throw the snake off entirely. For example, when we tried to reduce the turns the snake made by penalizing turns heavily, it would wound up making less turns but misses the apple.

[gifs here]

In the end, our final phase 1 reward system was:
- **+2** if the score increased after the chosen action
- **-1** if the snake agent ran into itself
- **-1.5** if the snake agent ran into a wall
- **-0.01** for every turn the snake took
- **+0.1** if the chosen action resulted in the snake agent getting closer to the fruit
- **-0.1** if the chosen action resulted in the snake agent getting farther from the fruit

<figure>
<video width="320" height="240" controls>
  <source src="./images/eval-episode-64.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
    <figcaption>Example of a run with the final reward system for phase 1.</figcaption>
</figure>

This updated reward system helps the agent know that the fruit is the main goal of the training process, resulting in the snake agent focusing on the fruit rather than finding a fast way reduce its penalty as much as possible.

### Phase 2
#### Observation Space

Our observation space for phase 2 consists of the following:
- "agent": snake head's current position on the grid (2D coordinate)
- "fruits": current positions and normalized values of all three fruits (regular fruit, decay fruit, enemy fruit) in the form: `[x, y, normalized_score_value]` for each fruit stored in a 2D array
- "danger": indicates whether moving in each cardinal direction (up, down, left, right) would result in collision with itself or a wall, using a dictionary of where dangers are based on direction
- **"enemy_danger"**: indicates whether the next move of the enemy fruit on its set square path would result in the snake eating the fruit

#### Reward System

For our phase 2 reward system, our initial reward system was the same as phase 1's final reward system.

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
- The snake agent ignored the regular fruit in many cases, chasing solely the decay fruit, even if it wouldn't reach the decay fruit in time 

[insert gifs here]

To improve learning, we introduced a distance-based shaping term that provides feedback on the snake’s progress toward the nearest fruit, encouraging efficient fruit collection. Distances are normalized using the Euclidean distance across the grid to ensure that reward changes are proportional, regardless of the snake’s position. The shaping term's formula is:

$$
S=0.5\times\frac{d_{prev} - d_{curr}}{\sqrt{L^2_x+L^2_y}}
$$

where  $d_{prev}$ and $d_{curr}$ are the distances between the snake head and the target fruit before and after it takes a step, and $L_x$ and $L_y$ are the board dimensions. Shaping adjustments were also applied based on each type of fruit so that the snake has the full context on which fruit would be the most helpful in score maximization.

Our final phase 2 reward system was:
- **+20** * $\times \Delta \text{score}$ on score increase
- **−15** * $\times \Delta \text{score}$ on score decrease
- **-12** if the game terminated (snake agent ran into a wall, itself, or died to an enemy fruit)
- **-0.01** for surviving
- **-0.003** for changing directions
- **-0.05** for moving away from any fruit
- **+0.5** * (normalized distance change to closest fruit) for feedback on snake progress towards fruits
- **+S** move toward regular fruit
- **-2S** move toward enemy fruit to heavily discourage it
- **+(1 + 2 * value_ratio) * S** for eating decay fruit. `value_ratio` is $\text{value\_ratio} = \frac{\text{current value of the fruit}}{\text{maximum value of the fruit}}$, to let the snake know the sooner it collects decay, the better.

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
- [ChatGPT](https://chatgpt.com/) used for tailoring out reward system when we were stuck. We gave it details of what undesirable behavior the snake agent was exhibiting and how we were rewarding the agent currently, and asked what we did not consider. This led to adding a small survival penalty and reducing/normalzing our distance reward so that it does not overwhelm our reward system. Additionally, we used ChatGPT to generate the necessary regular expression needed to parse slurm files to get training data for our evaluation.