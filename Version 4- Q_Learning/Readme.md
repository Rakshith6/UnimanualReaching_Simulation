** Q-Learning algorithm to update state action values**

_Files_
1. Run the **Parametric_Analysis.py** file. It runs 100 experiments for each pair of alfa and epsilon values and 
   outputs average espisodes to convergence and average experiment reward (reward of the final policy).
2. The Unimanual_Reaching_QLearning is imported as a module and runs the actual algorithm.
3. The Draw_3DHist is a module that I have created to draw any general 3D bar plot.

_Results_

1. The alfa and epsilon paramters are crucial towards determining average episodes to convergence.
2. Conducted a parametric analysis to identify that higher alfa values and lower epsilon values resulted in faster convergence.
   The paramteric analysis plot is attached for reference.
   
   ![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%204-%20Q_Learning/EpsilonAlfa_AvgEpisodeConvergence.png)
   
   (i) The fastest average episode to convergence was 40 (alfa = 0.8, epsilon = 0.05) which is much less than the previous versions. 
   (ii) In general the algorithm did much better than the previous adopted versions.
   
3. Majority of the experiments resulted in convergence to the most optimal solution of reaching in 6 time steps (The state evolution for this         optimal solution is attached in the repository). However, some episodes resulted in a suboptimal solution of reaching in 7 time steps, when using high alfa and low epsilon values.

*Optimal solution*

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%204-%20Q_Learning/Optimal_state_evolution.png)
  
