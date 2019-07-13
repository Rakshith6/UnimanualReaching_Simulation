# Introduction

Pitting random experience replay vs prioritized experience replay.

# Reward system

 - Altered the reward setting of the problem into a binary one. The agent gets a reward of +1 for getting to the target state else the reward is -1. The idea behind having such a reward setting is to avoid the design of a reward function, which might not be easy for complicated tasks. Moreover, we see the advantage of Prioritized Experience Replay clearly in binary reward systems. 
 - Each episode ended when the goal state was reached or when the episode exceeded 50 steps. Thus the minimum reward in any episode is equal to -50 and maximum reward is equal to -6 (optimal solution finsihes in 7 steps, check [optimal solution](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%204-%20Q_Learning/Optimal_state_evolution.png) )

# Learning setup

## Double DQN

The idea was derived from the following publication - [Link](https://arxiv.org/pdf/1509.06461.pdf) 

## Random experience replay

All transitions are stored in the memory without replacement. During replay a transition is chosen at random. Note that the probability of selection of transition depends proportionally on the frequency of that transition in the replay memory.

## Prioritized experience replay 

The idea is from the following publication - [Link](https://arxiv.org/abs/1511.05952)

### Explanation
 - The experience memory was populated with unique transitions (no repetition of transitions) only and a priority array was maintained for these transitions. 
 - A new incoming transition was initialized with max pirority so that it is replayed atleast once.
 - The transition was chosen using a Sum-tree data structure which samples transitions stochastically according to their priorities. Explanation of the sum tree in the following repo - [Link](https://github.com/Rakshith6/Sum-Tree)
 - The transition priority was updated using the absolute value of their TD errors.

# Results
The rewards were averaged across 30 runs and plotted with 1 standard deviations. The convergence rate for PER was around 70% and for random was close to 0.

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%206%20-%20DDQN%20%2B%20PER/RandomVsPER_DDQN.png)
 - Prioritized replay outperformed random replay.
 - Note that random replay did not lead to any significant learning even after 100 episodes.
