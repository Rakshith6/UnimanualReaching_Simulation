
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt


class neural_network:
    def __init__(self):
        self.state_no = 2
        self.action_no = 3

        self.create_network()

    def create_network(self):
        self.states = tf.placeholder(shape=[None,self.state_no],dtype=tf.float32)
        self.targets = tf.placeholder(shape=[None,self.action_no],dtype=tf.float32)
        self.layer_1 = tf.layers.dense(self.states,40,activation=tf.nn.relu)
        self.layer_2 = tf.layers.dense(self.layer_1, 40, activation=tf.nn.relu)
        self.Q_values = tf.layers.dense(self.layer_2,self.action_no)
        self.loss_fn = tf.losses.mean_squared_error(self.Q_values,self.targets)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss_fn)
        self.initialize_var = tf.global_variables_initializer()

    # returns forward propagations of the network - q-values
    def return_Qvalues(self,_states):
        return sess.run(self.Q_values,feed_dict = {self.states : _states})

    # backprop of batched randomly picked experience samples
    def train_samples(self,_states,_targets):
        return sess.run(self.optimizer, feed_dict={self.states:_states,self.targets :_targets})


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

            # if episode in np.arange(0,EPISODES,EPISODES/10):
            #     print('Running episode : {}'.format(episode))

            curr_state = np.zeros(2)
            next_state = np.zeros(2)
            self.episode_ended = False
            episode_reward = 0
            episode_states = []
            self.episode_steps = 0 # counting number of steps in the episode
            while True:
                episode_states.append(curr_state[0])
                action = self.determine_action(curr_state)
                reward,next_state=self.determine_transitions(curr_state,action)

                Exp_memory.save_transition([curr_state,action,reward,next_state])

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
        next_state[0]= curr_state[0] + curr_state[1] + 0.5 * action
        next_state[1] = curr_state[1] + action
        reward = -10
        if next_state[0] == 8 and next_state[1] == 0:
            reward = 1000
            self.episode_ended = True
        elif next_state[0]<-2 or next_state[0]>14:
            reward = -100

        return reward,next_state

    def determine_action(self,curr_state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.ACTIONS)
        else:
            return self.ACTIONS[np.argmax(NN.return_Qvalues(np.array(curr_state).reshape(1,2)))]

    def experience_replay(self):
        batch = Exp_memory.retrieve_batch()
        curr_states = np.array([I[0] for I in batch])
        next_states = np.array([np.zeros(2) if I[3] is None else I[3] for I in batch])
        q_values_current = NN.return_Qvalues(curr_states)
        q_values_next = NN.return_Qvalues(next_states)

        targets = np.zeros((len(batch),3))
        states = np.zeros((len(batch),2))

        for i,sample in enumerate(batch):
            curr_state,action,reward,next_state = sample[0],sample[1],sample[2],sample[3]

            target = q_values_current[i]

            if next_state is None:
                target[np.where(self.ACTIONS == action)] = reward
            else:
                target[np.where(self.ACTIONS == action)] = reward + GAMMA*np.max(q_values_next[i])

            targets[i] = target
            states[i] = curr_state

        NN.train_samples(states,targets)

    # running with learnt policy at the end of the run
    def run_learntPolicy(self):
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
            plt.figure(1000)
            pos_plot = plt.subplot(211)
            pos_plot.plot([I[0] for I in state_memory])
            pos_plot.set_yticks(np.arange(-1,12,1))
            pos_plot.set_ylabel('Position')

            vel_plot = plt.subplot(212)
            vel_plot.plot([I[1] for I in state_memory])
            vel_plot.set_ylabel('Velocity')
            vel_plot.set_xlabel('Steps')

        return success

class experience_memory:
    def __init__(self):
        self.memory = []
        self.memory_size = 1000
        self.batch_size = 50

    def save_transition(self,transition):
        self.memory.append(transition)
        if len(self.memory)>self.memory_size:
            self.memory.pop(0)

    def retrieve_batch(self):
        if len(self.memory)>self.batch_size:
            return random.sample(self.memory,self.batch_size)
        else:
            return self.memory

#hyperparameters
MAX_EPSILON = 0.99
MIN_EPSILON = 0.05
EPS_DECAY = 0.002
GAMMA = 0.9

EPISODES = 100
RUNS = 100
run_success = [] # stores whether the run resulted in a succesfully trained network
for run in range(RUNS):
    if run in np.arange(0,RUNS,RUNS/10):
        print('Running {}'.format(run))
    NN = neural_network()
    Sim = simulation()
    Exp_memory = experience_memory()
    sess = tf.Session()
    sess.run(NN.initialize_var)
    Sim.run_simulation()

    run_success.append(Sim.run_learntPolicy())

    if False:
        plt.figure(run)
        reward_plot = plt.subplot()
        reward_plot.plot(Sim.reward_memory)
        reward_plot.set_ylabel('Episode Reward')
        reward_plot.set_xlabel('Episode number')

    #plt.plot(np.cumsum(Sim.episode_outcome))

    sess.close()


