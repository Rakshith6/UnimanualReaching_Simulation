# UnimanualReaching_Simulation
Simulation of reaching with one hand in a straight line.

**Reaching Task**
1. The task is to move a the hand from start state (position, velocity) = (0,0) to a target state (position, velocity) = (8,0).
2. The hand chooses one of the three acceleration actions  (-1, 0, 1) at each step of movement.
3. At each step the action is chosen in an epsilon-greedy manner and receives a reward.
4. Reward of 100 is given upon reaching the target state, else a reward of 0  is awaded every other step.


**Experiment Protocol**

_Training_

Each 'learner' participates in an experiment and learns by repeating the task for a set number of trials (episodes). A policy is learnt based on updated state-action values after each trial.

_Post-test_

After the training, the learnt policy is put to test to see if it leads to a succesfull reach. If the reach is succesfull then the learner has learnt the right policy. 

**Parametric Testing**

Simulations were run for various combinations of exploration (epsilon = 0, 0.1, 0.5, 0.9, 1) and number of training episodes (episodes = 100, 500 ,1000) to find the most optimal training environement. 
100 learners (experiments) were tested for each combination of parameters.
The best combination of parameters will lead to the highest proportion of successfull learners (experiments which lead to succesfull reach in the post test). 

**Results**

1. There are several possible policies for succesfull learning. Evolution of states for two of those policies are shown in the example plots.
<p float="left">
  <img src="https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%201/StateEvolution_Episode1000Epsilon0.5_Example1.png" width="500" />
  <img src="https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%201/StateEvolution_Episode1000Epsilon0.5_Example2.png" width="500" /> 
</p>

2. Parametric testing revealed that epsilon = 0.5 and episodes = 1000 lead to most number of succesfull learning experiments (100%). Generally, epsilon of 0.5 performed the best and epsilon = 0 was by far the worst.
![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%201/EpsilonEpisode_ParamtericResults.png)
