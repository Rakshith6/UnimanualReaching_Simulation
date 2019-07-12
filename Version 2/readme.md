**Changes**
1. Altered the reward function to encourage reaching in minimum number of steps
2. Each episode now ends when the target state is reached or when the cursor goes beyond allowed positions. Previously, episodes would continue to run for set number of steps even if target state was reached.
3. Average rewards generated from the final policies of all experiments are evaluated for each epsilon-episode choice. Note that highest achievable reward is 995.

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%202/EpsilonEpisode_ParameterAvgReward_v2.png)

4. Based on results from 'Version 1' parametric analysis was conducted for the epsilon choices [0.1,0.3,0.4,0.5,0.6] and episode choices [100,300,400,500]

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%202/EpsilonEpisode_ParameterSuccessRate_v2.png)
