*Deep Q-network*
1. 2 fully connected layers with 40 nodes in each. ReLu activation for each layer and no activation for output (Q-values). Adam optimizer with learning rate of 0.005.
2. Epsilon greedy algorithm with exponentially decaying epsilon. Chose the decay rate based on the performance of agent in a single run.
3. Experience replay with random batch sampling

*Results*
1. Didnt perform better than tabular Q-learning (Version 4 converged at best in ~ 43 episodes on average)
2. Converged mostly to a suboptimal solution and the conergence rate was not very good (max number of episodes = 100)

