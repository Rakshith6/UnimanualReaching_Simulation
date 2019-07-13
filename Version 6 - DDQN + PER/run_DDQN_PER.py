
import numpy as np
import tensorflow as tf
import random
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Sum_Tree as ST
import pickle

class neural_network:
    def __init__(self):
        self.state_no = 2
        self.action_no = 3
        self.network = tf.keras.Sequential()
        self.create_network()

    def create_network(self):
        self.network.add(tf.keras.layers.Dense(40,input_shape=(self.state_no,),activation='relu'))
        self.network.add(tf.keras.layers.Dense(40,activation='relu'))
        self.network.add(tf.keras.layers.Dense(self.action_no))
        self.network.compile(optimizer=tf.keras.optimizers.Adam(),loss = tf.keras.losses.mean_squared_error)


    def return_Qvalues(self,_states):
        return self.network.predict_on_batch(_states)

    # backprop of batched randomly picked experience samples
    def train_samples(self,_states,_targets):
        return self.network.train_on_batch(_states,_targets)


class simulation:
    def __init__(self):
        self.ACTIONS = np.array([-1,0,1])
        self.reward = None
        self.episode_ended = False
        self.steps = 0 # counting steps in the entire run, used to exponentially decay epsilon
        self.state_memory = []
        self.reward_memory = []
        self.epsilon = MAX_EPSILON
        self.episode_outcome = np.zeros(EPISODES) # saves outcome of episode - 1 if agent was succesful at end of episode else 0

    # runs a single series of episodes
    def run_simulation(self):
        for episode in range(EPISODES):

            if episode in np.arange(0,EPISODES,EPISODES/10):
                print('Running episode : {}'.format(episode))

            curr_state = np.zeros(2)
            self.episode_ended = False
            episode_reward = 0
            episode_states = []
            self.episode_steps = 0 # counting number of steps in the episode
            while True:
                if self.steps%10 == 0:
                    Sim.copy_params()

                episode_states.append(curr_state[0])
                action = self.determine_action(curr_state)
                reward,next_state=self.determine_transitions(curr_state,action)

                transition = [curr_state.tolist(),action,reward,next_state.tolist()]

                Exp_memory.save_transition(transition)

                self.experience_replay()

                episode_reward+=reward

                curr_state = next_state

                self.steps += 1
                self.episode_steps+=1

                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-EPS_DECAY * self.steps)

                if self.episode_ended:
                    self.episode_outcome[episode] = 1

                if self.episode_ended or self.episode_steps>50:
                    self.reward_memory.append(episode_reward)
                    self.state_memory.append(episode_states)
                    break


    def determine_transitions(self,curr_state,action):
        next_state = np.zeros(2)
        next_state[0] = curr_state[0] + curr_state[1] + 0.5 * action
        next_state[1] = curr_state[1] + action
        reward = -1
        if next_state[0] == 8 and next_state[1] == 0:
            reward = 1
            self.episode_ended = True
        elif next_state[0]<-2 or next_state[0]>14:
            reward = -1

        return reward,next_state

    def determine_action(self,curr_state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.ACTIONS)
        else:
            return self.ACTIONS[np.argmax(NN.return_Qvalues(np.array(curr_state).reshape(1,2)))]

    def experience_replay(self):
        updates = Exp_memory.batch_size #if len(Exp_memory.memory)>=Exp_memory.batch_size else len(Exp_memory.memory)
        for update in range(updates):
            transition_index,transition = Exp_memory.retrieve_transition()
            curr_state,action,reward,next_state = np.array(transition[0]).reshape(1,2),transition[1],transition[2],np.array(transition[3]).reshape(1,2)

            q_values_current = NN.return_Qvalues(curr_state)
            q_values_next = NN.return_Qvalues(next_state)
            q_values_next_targetNN = Targ_NN.return_Qvalues(next_state)

            target = copy.copy(q_values_current)

            action_index  = np.where(self.ACTIONS == action)[0][0]

            max_action = np.argmax(q_values_next[0]) # max action selection from DQN network

            if next_state is None:
                target[0][action_index] = reward
            else:
                target[0][action_index] = reward + GAMMA*q_values_next_targetNN[0][max_action]

            td_error = target[0][action_index]-q_values_current[0][action_index]

            Exp_memory.update_transitionPriority(transition_index,td_error)

            NN.train_samples(curr_state,target)

    def copy_params(self):
        Targ_NN.network.set_weights(NN.network.get_weights())

    # running with learnt policy at the end of the run
    def run_learntPolicy(self,plot_no):
        curr_state = np.zeros(2)
        next_state = np.zeros(2)
        self.episode_ended = False
        self.steps = 0
        state_memory = []
        success = 0
        while True:
            state_memory.append(curr_state)
            action = self.ACTIONS[np.argmax(NN.return_Qvalues(np.array(curr_state).reshape(1,2)))]
            reward,next_state = self.determine_transitions(curr_state,action)

            curr_state = next_state
            self.steps += 1

            if self.episode_ended:
                success =1
            if self.episode_ended or self.steps>20:
                state_memory.append(curr_state)
                break

        # plotting the state outcomes of the learnt policy in an episode
        if False:
            plt.figure(plot_no)
            pos_plot = plt.subplot(211)
            pos_plot.plot([I[0] for I in state_memory])
            pos_plot.set_yticks(np.arange(-1,12,1))
            pos_plot.set_ylabel('Position')

            vel_plot = plt.subplot(212)
            vel_plot.plot([I[1] for I in state_memory])
            vel_plot.set_ylabel('Velocity')
            vel_plot.set_xlabel('Steps')

        return success

    def plot_Qvalues(self):
        position_samples = np.arange(-4,12,0.5)
        velocity_samples = np.arange(-2,2,0.5)

        state_pairs = np.array([[i,j] for i in position_samples for j in velocity_samples])

        Q_values = NN.return_Qvalues(state_pairs)

        for action_num,action in enumerate(self.ACTIONS):
            fig1 = plt.figure(action_num+100)
            ax = fig1.gca(projection='3d')
            Y, X = np.meshgrid(velocity_samples,position_samples)
            Z = np.array(Q_values[:,action_num]).reshape((len(position_samples),len(velocity_samples)))
            ax.plot_surface(X, Y, Z,shade=False,edgecolor='k',linewidth=0.5)


class experience_memory:
    def __init__(self):
        self.memory = []
        self.unique_memory = []
        self.memory_size = 10000
        self.batch_size = 50
        self.priorities = []
        self.transition_freq = np.zeros(10000)
        self.sum_tree = ST.Sum_Tree(self.memory_size)

    def save_transition(self,transition):
        self.memory.append(transition)

        if len(self.memory)>self.memory_size:
            self.memory.pop(0)

        if transition not in self.unique_memory:
            self.unique_memory.append(transition)
            new_priority = np.random.normal(1000,1)
            if replay_type == 'R':
                self.priorities.append(new_priority)
            elif replay_type == 'PER':
                self.sum_tree.priorities[len(self.unique_memory)-1] = new_priority
                self.sum_tree.update_leaf(len(self.unique_memory)-1,new_priority)

    def update_transitionPriority(self,index,td_error):
        new_priority = abs(td_error)
        if replay_type == 'R':
            self.priorities[index] = new_priority
        elif replay_type == 'PER':
            self.sum_tree.priorities[index] = new_priority
            self.sum_tree.update_leaf(index,new_priority)

    def retrieve_transition(self):
        if replay_type == 'R':
            index = np.random.choice(np.arange(0, len(self.unique_memory)))
        elif replay_type == 'PER':
            sum = np.random.uniform(0,self.sum_tree.tree_nodes[0])
            index,_ = self.sum_tree.get_priority(sum,0)
        self.transition_freq[index] += 1
        return index,self.unique_memory[index]


#hyperparameters
MAX_EPSILON = 0.99
MIN_EPSILON = 0.05
EPS_DECAY = 0.002
GAMMA = 0.9

EPISODES = 100
RUNS = 20
for I,replay_type in enumerate(['R']): # R = Random, PER = Prioritized experience replay
    run_success = [] # stores whether the run resulted in a succesfully trained network
    reward_memory = []
    for run in range(RUNS):
        if run in np.arange(0,RUNS,RUNS/10):
            print('Running {}'.format(run))
        NN = neural_network()
        Targ_NN = neural_network()

        Sim = simulation()
        Exp_memory = experience_memory()

        Sim.run_simulation()

        reward_memory.append(Sim.reward_memory)
        run_success.append(Sim.run_learntPolicy(run))

        if False:
            plt.figure(run+100)
            reward_plot = plt.subplot()
            reward_plot.plot(Sim.reward_memory)
            reward_plot.set_ylabel('Episode Reward')
            reward_plot.set_xlabel('Episode number')


# Storing reward data
store = True
if store:
    filename = 'Random_DDQN'
    outfile = open(filename,'wb')
    pickle.dump(reward_memory,outfile)
    outfile.close()

# Plots q-values for each action across state space
diagnostics = False
if diagnostics:
    Sim.plot_Qvalues()


