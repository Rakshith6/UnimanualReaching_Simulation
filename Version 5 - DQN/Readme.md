*Deep Q-network*
1. 2 fully connected layers with 40 nodes in each. ReLu activation for each layer and no activation for output (Q-values). Adam optimizer with learning rate of 0.005.
2. Epsilon greedy algorithm with exponentially decaying epsilon. Chose the decay rate based on the performance of agent in a single run.
3. Experience replay with random batch sampling
