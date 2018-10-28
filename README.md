# UnimanualReaching_Simulation
Simulation of reaching with one hand in a straight line.

**Reaching Task**
1. The task is to move a the hand from start state (position, velocity) = (0,0) to a target state (position, velocity) = (8,0).
2. The hand chooses one of the three acceleration actions  (-1, 0, 1) at each step of movement.
3. At each step the action is chosen in an epsilon-greedy manner and receives a reward.


**Experiment Protocol**

_Training_

Each 'learner' participates in an experiment and learns by repeating the task for a set number of trials (episodes). A policy is learnt based on updated state-action values after each trial.

_Post-test_

After the training, the learnt policy is put to test to see if it leads to a succesfull reach. If the reach is succesfull then the learner has learnt the right policy. 


