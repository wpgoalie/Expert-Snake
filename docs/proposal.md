---
layout: default
title:  Proposal
---
## Summary of the Project

The classic Snake Game traditionally consists of one snake collecting as many apples as it can without running into a wall or itself. At each point, a snake can choose to turn left or right by 90 degrees, or continue moving forward. The cheese variation of the game introduces a new level of complexity, as the snake's body is now split apart so that it only occupies every other tile. This allows the snake to move through itself through these open sections, but it also means that when the snake consumes an apple, it will grow by two tiles, one open secion and another its body. This will force the snake to have to maneuver through itself in order to achieve high scores, and also constrains where the apples can spawn since it cannot spawn within the body, making its spawning locations more predictable. 

Our project will focus on this implementation of the game, and we will aim to train an agent to efficiently navigate the board and utilize the gaps in its body in order to maximize the average score of the snake. We hope that the agent will leverage information about the snake, the possible fruit spawning locations, and the gaps in its body to score as much as possible and position itself in a way that allows for maximized fruit spawning.

## Project goals

* **Minimum Goal:** As a baseline goal, we will train a snake to achieve a score above that of an average player, which is 43-46 for a 9x10 board, 125-128 with a 15x17 board, and 250-253 with a 23x24 board.

* **Realistic goal:** For a realistic goal, we will try to maximize the score to be as close to the upper limit as possible, which based on research is 85 apples on a 9x10 board, 250 apples on a 15x17 board, and 499 apples on a 23x24 board. However, we recognize that the max scores listed are not always posisble and this is an estimate given ideal fruit spawning conditions. We will aim to get the average to be as close as possible.

* **Moonshot goal:** If we have the time, something that we do want to attempt is to turn this into a multi-agent game, where at least 2 or more snakes will compete with one another to gather as many apples as possible without crashing into the walls or each other. In this situation, we also expect to set rewards in such a way that the snake agents try to block each other off intentionally in order to destroy the other snake agent or stop them from getting more points through eating the apple before they can.

## AI/ML Algorithms

As the action space is discrete, the environment is stochastic, the horizon is episodic, and hyperparameters don't need to be constantly hypertuned, Proximal Policy Optimization (PPO) is the main algorithm we will be using on one of the snake agents, with Group Sequence Policy Optimization (GSPO) and Group Relative Policy Optimization (GRPO) also being similar algorithms we could use for the other snake agents. For the moonshot goal specifically, the minimax algorithm could also be potentially used to implement the opponent aspect between the snake agents.

## Evaluation Plan

In order to properly evaluate our agent quantitatively, we will run the agent within the cheese variation snake game and and have it play 1000 rounds across all three offered board sizes, and it will attempt to maximize its score. We will then take the average of its scores and use that as a metric to compare it against certain thresholds. As a baseline, we might implement a snake that chooses a direction at random to move in or a greedy snake that only takes the least amount of turns to get to the apple. The minimal goal is to beat average player data, which as mentioned above was 43-46 for a 9x10 board, 125-128 for a 15x17 board, and 250-253 with a 23x24 board. The main goal is to try and reach the theoretical max of the board as much as possible, meaning that we aim for 85 apples on a 9x10 board, 250 with 15x17, and 499 with 23x24. I think a reasonable ball park of how we can improve this metric is by roughly 50%.

In terms of comparing our outputs, we could find many different variations of a board being filled to its maximum possibility and see how our output grid differs in regard to the snake's position and its gaps in its body. For sanity cases, we could test cases of where the snake going through the gaps is more optimal than avoiding the gaps and see if the agents go with the more optimal choice. For multi-agents, we could visualize the grid and see if the snake agents use the gaps to sneakily trap off the other snakes or potentially destroy them. For debugging, some cases we should check would be to see if the gaps in the body are truly registering as open space by the agents or if the agents still classify these gaps as part of the snake's body.

## AI Tool Usage

Potential algorithm ideas were provided by ChatGPT, and we picked a few to research in order to decide if they were viable or not, including the Group Sequence Policy Optimization (GSPO) and Group Relative Policy Optimization (GRPO) algorithms.

---
## Summary of the Project

The classic Snake Game traditionally consists of one snake collecting as many apples as it can without running into a wall or itself. The action space is {up, down, left, and right}, with one of those actions being impossible to do based on the direction of the snake, as you can't turn back into yourself 180 degrees. In our project, we decided to implement two to three snake agents in the grid environment instead of just one snake agent, each running its own algorithm out of this list: Policy Optimization (PPO), Group Sequence Policy Optimization (GSPO), and Group Relative Policy Optimization (GRPO). The snake agents are playing against each other in our project, where they are trying to maximize their score and minimize their opponents' scores, which is what our project analysis will mainly focus on.

## Project goals

* **Minimum Goal:** Implementing a two-agent Snake game where the agents are against each other would be the minimum amount to get finished in this project. For this goal, we expect for both snake agents to reach a length of at least 1/8th of the area of the grid each while still staying alive.

* **Realistic goal:** Implementing a three-agent Snake game where the agents are against each other would be a realistic amount of work we could finish for this project. For this goal, we expect each snake agent to reach a length of at least 1/6th of the area of the grid while still staying alive. We also expect to set rewards in such a way that the snake agents try to block each other off intentionally in order to destroy another snake agent or stop them from getting more points through eating apples.

* **Moonshot goal:** Modifying the grid so that snakes are able to go through walls and teleport to the wall opposite of the wall they went through would be a moonshot modification we will keep in mind if time allows, as this modification of the grid will greatly change the strategies the snake agents would have to use in order to stay alive and try to sabotage the other snake agents.

## AI/ML Algorithms

As the action space is discrete, the environment is stochastic, the horizon is episodic, and hyperparameters don't need to be constantly hypertuned, Proximal Policy Optimization (PPO) is the main algorithm we will be using on one of the snake agents, with Group Sequence Policy Optimization (GSPO) and Group Relative Policy Optimization (GRPO) also being similar algorithms we could use for the other snake agents. The minimax algorithm could also be potentially used to implement the opponent aspect between the snake agents.

## Evaluation Plan

Initially, we will implement the Proximal Policy Optimization (PPO), Policy Optimization (GSPO), and Group Relative Policy Optimization (GRPO) algorithms all separately in a single-agent system, where the goal is for each snake to collect one fourth of the apples on the grid (ex.: 25 apples on a 10x10 grid). We will then have our team members play five rounds of the Snake game each and get the average score between all fifteen total rounds. Once we get to a point where all agents beat our average score by around 25-35% we will implement a Minimax algorithm in order to have the agents go against each other in a multi-agent system.

In terms of toy cases, we plan to scale the grid and analyze how the behavior of each agent cahnges, scaling the grid from an initial start of a 5x5 grid all the way up to a 25x25 grid. We plan to display the board in a virtual UI screen and constantly show the real-time states of each snake agent and the fruits to collect. We also plan to show the current score of each agent currently on the board in order to compare the scores, and even potentially show the time at which a snake agent ran into a wall or got destroyed by another snake agent. We will know that we successfully implemented the algorithms and tuned their hyperparameters correctly if they consistently go to the apples on the grid in an efficient path while also actively rerouting their path to avoid the other snake agents or even to cut the other agents off to try to trap/destroy them.

## AI Tool Usage

ChatGPT was used for potential gameplay modification aspects to add to the Snake game, such as being able to teleport through walls. Potential algorithm ideas were also provided by ChatGPT, in which we researched some of these algorithm ideas through articles in order to decide if these algorithms were viable or not.
