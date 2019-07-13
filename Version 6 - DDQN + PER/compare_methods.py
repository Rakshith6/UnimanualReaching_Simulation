import matplotlib.pyplot as plt
import pickle
import numpy as np

file_list = ['Random_DDQN','PER_DDQN']

plt.figure(1)
reward_plot = plt.subplot()
reward_plot.set_xlabel('Episodes')
reward_plot.set_ylabel('Average reward (30 runs)')

for file_name in file_list:
    openfile = open(file_name,'rb')
    reward_memory = pickle.load(openfile)
    openfile.close()

    mean = np.mean(reward_memory,axis = 0)
    stdev = np.std(reward_memory,axis=0)
    episodes = np.arange(1,101)
    reward_plot.plot(episodes,mean,label = file_name)
    reward_plot.fill_between(episodes,mean+stdev,mean-stdev,alpha = 0.5)
reward_plot.legend()
