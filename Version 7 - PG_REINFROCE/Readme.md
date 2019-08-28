# Policy gradients  - REINFORCE

## Neural network
* Two fully connected hidden layers with ReLU activation. 
* Softmax activation on the logits layer to obtain action probabilities.
* Adam optimizer with - loss function = softmax cross entropy on action probabilities * advantage

    Here the cross entropy was computed with the selected actions as true labels and advantage values were the discounted       rewards. 


## Results

* The algorithm had a convergence rate of 0.86 from running 50 learning runs. 
* As expected, many of the runs converged to the sub optimal solution as shown below.

![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%207%20-%20PG_REINFROCE/Sub-optimal_state_evolution.png)

* The average reward and loss from 50 runs is shown below. The maximum reward possible is 940. Used the same reward function from the first few versions.
![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%207%20-%20PG_REINFROCE/average_reward.png) ![](https://github.com/Rakshith6/UnimanualReaching_Simulation/blob/master/Version%207%20-%20PG_REINFROCE/average_loss.png)



