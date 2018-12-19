import Unimanual_Reaching_Qlearning as run_sim
import Draw_3DHist as draw_hist
import numpy as np
import pickle

ALFA_VALUES = [0.05,0.1,0.5,0.8]
EPSILON_VALUES = [0.05,0.1,0.2]

reward_results = np.zeros((len(EPSILON_VALUES),len(ALFA_VALUES)))
episodenumber_results = np.zeros((len(EPSILON_VALUES),len(ALFA_VALUES)))
for i,EPSILON in enumerate(EPSILON_VALUES):
    for j,ALFA in enumerate(ALFA_VALUES):
        episodenumber_results[i, j],reward_results[i,j]= run_sim.run_experiments(ALFA,EPSILON)

# f = open('AvgEps_results.pckl', 'wb')
# pickle.dump(episodenumber_results, f)
# f.close()
# f = open('AvgReward_results.pckl', 'wb')
# pickle.dump(reward_results, f)
# f.close()

#Use pickle to load data
#f = open('AvgEps_results.pckl', 'rb')
#episodenumber_results = pickle.load(f)

ax = draw_hist.draw_plot(EPSILON_VALUES,ALFA_VALUES,episodenumber_results,0.05,0.05)

ax.set_xlabel('Epsilon')
ax.set_ylabel('Alfa')
ax.set_zlabel('Average Episodes to Converge')

