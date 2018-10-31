import numpy as np
import matplotlib.pyplot as plt
import copy as copy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pickle

def choose_action(pos,vel):
    if not Q_values:
        return np.random.choice(actions)
    else:
        choice_list = []
        value_list = []
        for I in Q_values:
            if I[0][0] == pos and I[0][1] == vel:
                choice_list.append(I)
                value_list.append(I[2])

        if not choice_list:
            return np.random.choice(actions)
        else:
            e = np.random.random()
            if e<epsilon:
                return np.random.choice(actions)
            else:
                index = np.random.choice(np.flatnonzero(value_list == np.max(value_list)))
                return choice_list[index][1]
def best_action(pos,vel):
    choice_list = []
    action_value = []
    for I,V in enumerate(Q_values):
        if V[0][0]== pos and V[0][1] == vel:
            choice_list.append(V)
            action_value.append(V[2])

    if not choice_list:
        return np.random.choice(actions)
    else:
        index = np.argmax(action_value)
        return choice_list[index][1]

def determine_state(action,pos,vel):
    next_pos = pos + vel+0.5*action
    next_vel = vel + action
    return next_pos,next_vel

def determine_reward(pos,vel):
    if pos == 8 and vel == 0:
        reward = 1000
    elif pos < 0 or pos > 10.0:
        reward = -100
    else:
        reward = -1

    return reward

def update_Q_values(H):
    discount = 0.9
    G = 0;
    for step in np.arange(len(H) - 3, -1, -3):
        G = discount * G +  H[step + 2]

        first_visit = True
        check_step = step
        while check_step > 0 and first_visit:
            check_step = check_step - 3
            if H[check_step][0] == H[step][0] and H[check_step][1] == H[step][1] and H[check_step + 1] == H[step+1]:
                first_visit = False
        if first_visit:
            stateaction_exists = False
            for index,V in enumerate(Q_values):
                if V[0][0] == H[step][0] and V[0][1] == H[step][1] and V[1] == H[step+1]:
                    stateaction_exists = True
                    Q_values[index][3]+=1
                    Q_values[index][2] = Q_values[index][2] + (G - Q_values[index][2])/Q_values[index][3]

            if not stateaction_exists:
                Q_values.append([H[step],H[step + 1],G,1])

def simulate_learnt_policy():
    current_pos = 0
    current_vel = 0
    pos = []
    vel = []
    step = 0
    converged = False
    cumulative_reward = 0
    while current_pos > -2 and current_pos < 12 and step < 20 and not converged:
        pos.append(current_pos)
        vel.append(current_vel)
        action = best_action(current_pos, current_vel)
        current_pos, current_vel = determine_state(action, current_pos, current_vel)
        cumulative_reward = cumulative_reward+determine_reward(current_pos, current_vel)
        step += 1
        if current_pos == 8 and current_vel == 0 and not converged:
            converged = True
            pos.append(current_pos)
            vel.append(current_vel)
        elif not step < 20:
            pos.append(current_pos)
            vel.append(current_vel)

    return cumulative_reward

actions = [-1,0,1]

# Stores values in the order [epsilon, episodes, proportion of success, average steps to success, average experiment reward]


Experiments = 10000
Experiment_result = [[0,0] for i in range(Experiments)]

PlotNo = 0
for Experiment in range(Experiments):
    Q_values = []
    if Experiment in np.arange(0,Experiments,Experiments/10):
        print("Running experiments number {}".format(Experiment))

    episode_reward = []
    episode = 0
    epsilon = 0.5
    converging = False
    converge_count = 0
    while converge_count < 20 and episode < 500:
        current_pos = 0
        current_vel = 0
        history = []
        final_reward = 0
        for step in np.arange(1,21):
            history.append([current_pos,current_vel])
            action = choose_action(current_pos,current_vel)
            history.append(action)
            current_pos,current_vel = determine_state(action,current_pos,current_vel)
            reward = determine_reward(current_pos,current_vel)
            history.append(reward)
            if current_pos <= -2 or current_pos >= 12:
                break
            if current_pos == 8 and current_vel == 0:
                break

        update_Q_values(history)
        final_reward = simulate_learnt_policy()
        episode_reward.append(final_reward)
        if episode_reward[episode]>900 and episode_reward[episode]<994:
            converged_episode = episode
            converging = True
            epsilon = 0.5
        elif episode_reward[episode] >= 994:
            converged_episode = episode
            epsilon = 0.5
            converge_count+=1
        episode+=1

    Experiment_result[Experiment][0] = converged_episode
    Experiment_result[Experiment][1] = episode_reward[episode-1]


    plot_reward = False
    if plot_reward == True and converged_episode>500 and PlotNo < 15:
        plt.figure(Experiment)
        reward_plot = plt.subplot()
        reward_plot.plot(np.arange(episode),episode_reward,'r')
        reward_plot.set_yticks(np.arange(-400, 1000, 100))
        reward_plot.set_xlabel('Episode number')
        reward_plot.set_ylabel('Reward')

        plt.draw()
        plt.show(block = False)
        PlotNo+=1


if plot_reward == True:
    plt.show()

final_results = True
if final_results:
    avg_reward = []
    avg_episode = []
    for I,V in enumerate(Experiment_result):
        avg_reward.append(V[1])
        avg_episode.append(V[0])

    plt.figure(figsize=(12,6))
    histplot_ep = plt.subplot(121)
    histplot_ep.hist(avg_episode,bins = 20)
    histplot_ep.set_ylim([0, 3000])
    histplot_ep.set_xlabel('Episodes to convergence')
    histplot_ep.set_ylabel('Number of experiments')

    histplot_rd = plt.subplot(122)
    histplot_rd.hist(avg_reward, bins=range(990,997))
    histplot_rd.set_ylim([0,10000])
    histplot_rd.set_xlabel('Cumulated reward from final policy')
    histplot_rd.set_ylabel('Number of experiments')

    plt.savefig('Epsilon0.5-0.5-0.5.png')

    plt.show()

print('avg reward {} and avg episode {}'.format(np.mean(avg_reward),np.mean(avg_episode)))

