import numpy as np
import matplotlib.pyplot as plt

def choose_action(pos,vel,policy):
    state_exists = False
    for index,I in enumerate(Q_values):
        if I[0] == pos and I[1] == vel:
            max_index = np.random.choice(np.flatnonzero(I[2]==np.max(I[2])))
            state_exists = True
            state_index = index
            break

    if state_exists:
        if policy == 'e':
            e = np.random.random()
            if e<EPSILON:
                action  = np.random.choice(ACTIONS)
            else:
                action = ACTIONS[max_index]
        else:
            action = ACTIONS[max_index]
    else:
        action = np.random.choice(ACTIONS)
        state_index = len(Q_values)-1

    return action,state_index

def determine_state(pos,vel,action):
    next_pos = pos + vel+0.5*action
    next_vel = vel + action
    if next_pos == 8 and next_vel == 0:
        reward = 10
    elif next_pos <0 or next_pos>10:
        reward = -10
    else:
        reward = -1
    return next_pos,next_vel,reward

def update_Qvalues(p0,v0,action,reward,p1,v1,state_index):
    state_exists = False
    action_index = int(np.flatnonzero(ACTIONS == action))
    for I in Q_values:
        if I[0] == p1 and I[1] == v1:
            max_Qvalue = np.max(I[2])
            state_exists = True
            break
    if not state_exists:
        Q_values.append([p1,v1,[0,0,0]])
        Q_values[state_index][2][action_index] = Q_values[state_index][2][action_index] + ALFA * (
                    reward  - Q_values[state_index][2][action_index])
    else:
        Q_values[state_index][2][action_index] = Q_values[state_index][2][action_index] + ALFA*(reward + GAMMA*max_Qvalue-Q_values[state_index][2][action_index])


def run_experiment():

    EPISODES = 200
    EPISODE_REWARDS = np.zeros(EPISODES)
    converged_episode = 0
    converged_episodes_count = 0
    for episode in range(EPISODES):
        current_pos = 0
        current_vel = 0
        step = 0
        converged = False
        cumulative_reward = 0
        while not converged and current_pos>=0 and current_pos < 10:
            action,current_state_index = choose_action(current_pos, current_vel,'e')
            new_pos, new_vel,reward = determine_state( current_pos, current_vel,action)
            update_Qvalues(current_pos,current_vel,action,reward,new_pos,new_vel,current_state_index)
            current_pos = new_pos
            current_vel = new_vel
            cumulative_reward +=reward
            step += 1
            if current_pos == 8 and current_vel == 0:
                if converged_episode == 0:
                    converged_episode = episode
                converged = True

        current_pos = 0
        current_vel = 0
        pos = []
        vel = []
        step = 0
        converged = False
        cumulative_reward = 0
        while step<30:
            pos.append(current_pos)
            vel.append(current_vel)
            action,current_state_index = choose_action(current_pos, current_vel,'g')
            new_pos, new_vel,reward = determine_state( current_pos, current_vel,action)
            current_pos = new_pos
            current_vel = new_vel
            cumulative_reward +=reward
            step+=1
            if current_pos == 8 and current_vel == 0:
                pos.append(current_pos)
                vel.append(current_vel)
                break

        if pos[-1] == 8 and vel[-1] == 0:
            converged_episodes_count += 1
        else:
            converged_episodes_count = 0

        if converged_episodes_count >10:
            break

        EPISODE_REWARDS[episode] = cumulative_reward


    state_plot = True
    if state_plot:
        plt.figure(1)
        Pos_Plot = plt.subplot(2,1,1)
        Pos_Plot.plot(np.arange(0,step+1,1),pos)
        Pos_Plot.set_xticks(np.arange(0,10,1))
        Pos_Plot.set_yticks(np.arange(0, 10, 1))
        Pos_Plot.set_xlabel('Step number (time)')
        Pos_Plot.set_ylabel('Position')

        Vel_Plot =plt.subplot(2,1,2)
        Vel_Plot.plot(np.arange(0,step+1,1),vel)
        Vel_Plot.set_xticks(np.arange(0,10,1))
        Vel_Plot.set_yticks(np.arange(0, 4, 1))
        Vel_Plot.set_xlabel('Step number (time)')
        Vel_Plot.set_ylabel('Velocity')

        plt.show()

    reward_plot = False
    if reward_plot:
        plt.figure(2)
        Reward_Plot = plt.subplot()
        Reward_Plot.plot(np.arange(EPISODES),EPISODE_REWARDS)

    return episode - 10, cumulative_reward

def run_experiments(alfa,epsilon):
    global ACTIONS, ALFA, EPSILON, GAMMA, Q_values
    ALFA = alfa
    EPSILON = epsilon
    ACTIONS = np.array([-1,0,1])
    GAMMA = 1
    EXPERIMENTS = 1

    print('Running epsilon = {} and alfa = {}'.format(EPSILON,ALFA))
    experiment_rewards = np.zeros(EXPERIMENTS)
    experiment_episodes = np.zeros(EXPERIMENTS)
    for experiment in range(EXPERIMENTS):
        if experiment in np.arange(0,EXPERIMENTS,EXPERIMENTS/10):
            print('Running experiment {}'.format(experiment))
        Q_values = [[0,0,[0,0,0]],[8,0,[0,0,0]]]
        experiment_episodes[experiment],experiment_rewards[experiment] = run_experiment()

    return experiment_episodes.mean(),experiment_rewards.mean()

