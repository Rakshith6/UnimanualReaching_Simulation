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


def determine_state(action,pos,vel):
    next_pos = pos + vel+0.5*action
    next_vel = vel + action
    return next_pos,next_vel

def determine_reward(pos,vel):
    if pos == 8 and vel == 0:
        reward = 100
    else:
        reward = 0

    return reward

def update_Q_values(H):
    G = 0;
    for step in np.arange(len(H) - 3, -1, -3):
        G = G + H[step + 2]
        stateaction_exists = False
        for index,V in enumerate(Q_values):
            if V[0][0] == H[step][0] and V[0][1] == H[step][1] and V[1] == H[step+1]:
                first_visit = True
                stateaction_exists = True
                check_step = step
                while check_step > 0 and first_visit:
                    check_step = check_step - 3
                    if H[check_step][0] == V[0][0] and H[check_step][1] == V[0][1] and H[check_step + 1] == V[1]:
                        first_visit = False
                if first_visit:
                    Q_values[index][3]+=1
                    Q_values[index][2] = Q_values[index][2] + (G - Q_values[index][2])/Q_values[index][3]
                break
        if not stateaction_exists:
            Q_values.append([H[step],H[step + 1],G,1])


YPos_lim = [0,10]
YVel_lim = [-10,10]

actions = [-1,0,1]

episode_epsilon_choice = [[i,j,0] for i in [0.5] for j in [1000]]

for epsepi_choice,run_condition in enumerate(episode_epsilon_choice):
    Episodes = run_condition[1]
    epsilon = run_condition[0]

    print("Running experiments for Epsilon {} and Episode {}".format(epsilon, Episodes))

    Experiments = 10

    Experiment_result = 0

    for Experiment in range(Experiments):
        Q_values = []

        for episode in range(Episodes):
            current_pos = 0
            current_vel = 0
            history = []
            for step in np.arange(1,21):
                history.append([round(current_pos,1),round(current_vel,1)])
                action = choose_action(round(current_pos,1),round(current_vel,1))
                history.append(action)
                current_pos,current_vel = determine_state(action,current_pos,current_vel)
                reward = determine_reward(round(current_pos,1),round(current_vel,1))
                history.append(reward)
                if current_pos < -2 or current_pos > 12:
                    break


            update_Q_values(history)

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


        # Simulating the best policy
        current_pos = 0
        current_vel = 0
        pos_data = []
        vel_data = []
        step=0
        while current_pos>-2 and current_pos < 14 and step<20:
            pos_data.append(current_pos)
            vel_data.append(current_vel)
            action = best_action(round(current_pos,1),round(current_vel,1))
            current_pos,current_vel = determine_state(action,current_pos,current_vel)
            step+=1

        if current_vel == 0 and current_pos == 8:
            Experiment_result+=1



        # if Experiment==0:
        #     AllPos_data = copy.copy(pos_data)
        #     AllVel_data = copy.copy(vel_data)
        # else:
        #     AllPos_data = np.vstack((AllPos_data,pos_data))
        #     AllVel_data = np.vstack((AllVel_data, vel_data))

        animate = False
        if animate == True:
            fig = plt.figure()
            Plot = plt.subplot()
            Plot.set_xlim([-1,1])
            Plot.set_ylim([-2,14])
            Plot.set_yticks(np.arange(-2,15,1))

            Plot.plot(0,8,'go',markersize = 12)
            Plot.plot(0, 0, 'ko', markersize=12)
            state = Plot.plot(0,pos_data[0],'ro')

            def draw_frame(i):
                state[0].set_data(0,pos_data[i])

            ani = FuncAnimation(fig,draw_frame,interval = 1000, frames = len(pos_data) - 1)

            plt.draw()
            plt.show()

        state_plot = True
        if state_plot:
            plt.figure(Experiment)
            Pos_Plot = plt.subplot(2,1,1)
            Pos_Plot.plot(range(step),pos_data)
            Pos_Plot.set_xticks(np.arange(0,step,1))
            Pos_Plot.set_xlabel('Step number (time)')
            Pos_Plot.set_ylabel('Position')

            Vel_Plot =plt.subplot(2,1,2)
            Vel_Plot.plot(range(step),vel_data)
            Vel_Plot.set_xticks(np.arange(0,step,1))
            Vel_Plot.set_xlabel('Step number (time)')
            Vel_Plot.set_ylabel('Velocity')

            plt.show()

    episode_epsilon_choice[epsepi_choice][2] = Experiment_result / Experiments

f = open('Results.pckl', 'wb')
pickle.dump(episode_epsilon_choice, f)
f.close()

k=0
Z = np.zeros((5,3))
for i in range(5):
    for j in range(3):
        Z[i,j] = episode_epsilon_choice[k][2]*100
        k+=1

fig2 = plt.figure(figsize = (10,5))
ax = fig2.gca(projection='3d')
_X = [0,0.1,0.5,0.9,1.0]
_Y= [ 100, 500 ,1000]
YY,XX=np.meshgrid(_Y,_X)
X, Y = XX.ravel(), YY.ravel()
Z = Z.ravel()
bottom = np.zeros_like(Z)
dx = 0.06
dy = 100

ax.bar3d(X,Y,bottom,dx,dy,Z,shade=True)

ax.set_xticks(np.add([0,0.1,0.5,0.9,1.0],dx/2))
ax.set_xticklabels(['0','0.1','0.5','0.9','1.0'])
ax.set_yticks(np.add([100, 500 ,1000],dy/2))
ax.set_yticklabels(['100', '500' ,'1000'])
ax.set_zticks(np.arange(0,101,20))
ax.set_xlabel('Epsilon (exploration)')
ax.set_ylabel('Number of episodes')
ax.set_zlabel('Success rate (%)')
plt.savefig("EpsilonEpisode_ParamtericResults.png")
