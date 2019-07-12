
The results from version two suggested that an epsilon of 0.5 would lead to high rates of convergence. 
The goal of this version was to adapt exploration so that the learner takes minimum number of episodes to converge to a desired reward.

**Stagewise Exploration**

Stage 1: The exploration starts with epsilon = 0.5 as long as cumulative reward in an episode is small (<900).

Stage 2: Once the learner achieves a cumulative reward > 900 and <=993 i.e. the learner has identified the target state. Continue exploration to find policy that generates reward >= 994. 

Stage 3: Starts when the learner achieves a reward >= 994. 

_Exploration value settings (Stage 1 > Stage 2 > Stage 3)_

Exploration at each stage under four types as listed below:

Type 1: (0.5 > 0.5 > 0.5)

Type 2: (0.5 > 0.5 > 0.3)

Type 3: (0.5 > 0.5 > 0.1)

Type 4: (0.5 > 0.3 > 0.1)

**Results**

For each Type of exploration, 10000 experiments were conducted to find distribution and average behaviors. Each experiment was terminated at 500 episodes irrespective of reaching stage 2 or 3:
1. Histograms of episodes to convergence and final policy cumulative reward were constructed.
2. The average episodes to convergence and episode reward were calculated.

Type 1: Avg episode to convergence = 179.3, Avg Reward= 993.47 

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%203-Stagewise%20exploration/Epsilon0.5-0.5-0.5.png)

Type 2: 161.5, 993.69

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%203-Stagewise%20exploration/Epsilon0.5-0.5-0.3.png)

Type 3: 145.02, 992.99

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%203-Stagewise%20exploration/Epsilon0.5-0.5-0.1.png)

Type 4: 195.10, 992.98

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%203-Stagewise%20exploration/Epsilon0.5-0.3-0.1.png)
