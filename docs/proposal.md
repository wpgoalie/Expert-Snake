---
layout: default
title:  Proposal
---
## Summary of the Project

The classic Snake Game traditionally consists of one snake collecting as many apples as it can without running into a wall or itself. The action space is {up, down, left, and right}, with one of those actions being impossible to do based on the direction of the snake, as you can't turn back into yourself 180 degrees. In our project, we decided to implement two to three snake agents in the grid environment instead of just one snake agent, each running its own algorithm out of this list: Policy Optimization (PPO), Group Sequence Policy Optimization (GSPO), and the Group Relative Policy Optimization (GRPO). The snake agents are playing against each other in our project, where they are trying to maximize their score and minimize their opponents' scores, which is what our project analysis will mainly focus on.

## Project goals

* **Minimum Goal:** Implementing a two-agent Snake game where the agents are against each other would be the minimum amount to get finished in this project. For this goal, we expect for both snake agents to reach a length of at least 1/8th of the area of the grid each while still staying alive.

* **Realistic goal:** Implementing a three-agent Snake game where the agents are against each other would be a realistic amount of work we could finish for this project. For this goal, we expect each snake agent to reach a length of at least 1/6th of the area of the grid while still staying alive. We also expect to set rewards in such a way that the snake agents try to block each other off intentionally in order to kill another snake agent or stop them from getting more points through eating apples.

* **Moonshot goal:** Modifying the grid so that snakes are able to go through walls and teleport to the wall opposite of the wall they went through would be a moonshot modification we will keep in mind if time allows, as this modification of the grid will greatly change the strategies the snake agents would have to use in order to stay alive and try to sabotage the other snake agents.

## AI/ML Algorithms

## Evaluation Plan

## AI Tool Usage