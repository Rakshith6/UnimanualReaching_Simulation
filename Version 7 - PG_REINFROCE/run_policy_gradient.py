import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pickle

class neural_network:
    def __init__(self):
        self.state_no = 2
        self.action_no = 3
        self.states = tf.placeholder(tf.float32,shape = (None,self.state_no))
        self.episode_actions = tf.placeholder(tf.int32,shape = (None))
        self.advantages  = tf.placeholder(tf.float32,shape = (None))
        self.create_network()

    def create_network(self):
        self.fc1 = tf.layers.Dense(40,activation='relu')(self.states)
        self.fc2 = tf.layers.Dense(40,activation='relu')(self.fc1)
        self.logits = tf.layers.Dense(self.action_no)(self.fc2)
        self.action_probs = tf.keras.layers.Activation('softmax')(self.logits)

        self.loss= tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels = self.episode_actions)*self.advantages)

        self.train = tf.train.AdamOptimizer().minimize(self.loss)

class simulation:
    def __init__(self):
        self.ACTIONS = np.array([-1,0,1])
        self.episode_ended = False
        self.steps = 0 # counting steps in the entire run, used to exponentially decay epsilon
        self.state_memory = []
        self.reward_memory = []
        self.loss_memory = np.zeros(EPISODES)
        self.episode_outcome = np.zeros(EPISODES) # saves outcome of episode : 1 if agent was succesful at end of episode else 0

    # runs a single series of episodes
    def run_simulation(self):
        for episode in range(EPISODES):
            if episode in np.arange(0,EPISODES,EPISODES/10):
                print('Running episode : {}'.format(episode))

            curr_state = np.zeros(2)
            self.episode_ended = False
            episode_total_reward = 0
            episode_rewards,episode_states,episode_actions = [],[],[]
            self.episode_steps = 0 # counting number of steps in the episode

            transition_list = []
            while True:
                action = self.determine_action(curr_state)
                reward,next_state=self.determine_transitions(curr_state,action)


                episode_states.append(curr_state)
                episode_actions.append(np.where(self.ACTIONS==action)[0][0])
                episode_rewards.append(reward)

                episode_total_reward+=reward

                curr_state = next_state

                self.steps += 1
                self.episode_steps+=1

                if self.episode_ended:
                    self.episode_outcome[episode] = 1

                if self.episode_ended or self.episode_steps>50:
                    discounted_rewards = self.determine_discounted_rewards(episode_rewards)

                    self.loss_memory[episode], _ = sess.run([NN.loss, NN.train],
                                                            feed_dict={NN.states: episode_states,
                                                                       NN.episode_actions: episode_actions,
                                                                       NN.advantages: discounted_rewards})

                    self.reward_memory.append(episode_total_reward)
                    self.state_memory.append(episode_states)
                    break


    def determine_transitions(self,curr_state,action):
        next_state = np.zeros(2)
        next_state[0] = curr_state[0] + curr_state[1] + 0.5 * action
        next_state[1] = curr_state[1] + action
        reward = -10
        if next_state[0] == 8 and next_state[1] == 0:
            reward = 1000
            self.episode_ended = True
        elif next_state[0]<-2 or next_state[0]>14:
            reward = -100

        return reward,next_state

    def determine_action(self,curr_state):
        action_probs = sess.run(NN.action_probs, feed_dict={NN.states : curr_state.reshape(1,2)})

        return np.random.choice(self.ACTIONS,p = action_probs[0])

    def determine_discounted_rewards(self,rewards):
        G = 0
        discounted_rewards = np.zeros(len(rewards))
        for i,r in enumerate(reversed(rewards)):
            G = r + GAMMA * G
            discounted_rewards[i] = G

        discounted_rewards/= np.std(discounted_rewards)
        return discounted_rewards

    def run_learntPolicy(self,plot_no):
        curr_state = np.zeros(2)
        next_state = np.zeros(2)
        self.episode_ended = False
        self.steps = 0
        state_memory = []
        success = 0
        while True:
            state_memory.append(curr_state)
            action = self.ACTIONS[np.argmax(sess.run(NN.action_probs, feed_dict={NN.states : curr_state.reshape(1,2)}))]
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
            plt.figure(plot_no+100)
            pos_plot = plt.subplot(211)
            pos_plot.plot([I[0] for I in state_memory])
            pos_plot.set_yticks(np.arange(-1,12,1))
            pos_plot.set_ylabel('Position')

            vel_plot = plt.subplot(212)
            vel_plot.plot([I[1] for I in state_memory])
            vel_plot.set_ylabel('Velocity')
            vel_plot.set_xlabel('Steps')

        return success

EPISODES = 2000
GAMMA = 0.99
RUNS = 50

reward_memory = []
loss_memory = []
run_success = []
for run in range(RUNS):
    if run in np.arange(0, RUNS, RUNS / 10):
        print('Running {}'.format(run))
    NN = neural_network()
    Sim = simulation()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    Sim.run_simulation()

    run_success.append(Sim.run_learntPolicy(run))
    reward_memory.append(Sim.reward_memory)
    loss_memory.append(Sim.loss_memory)

    Sim.run_learntPolicy(run)

    # plt.figure(run*2)
    # plt.plot(Sim.loss_memory)
    #
    # plt.figure(run*2+1)
    # plt.plot(Sim.reward_memory)

    sess.close()

plt.figure(0)
Reward_plot = plt.subplot()
Reward_plot.plot(np.mean(reward_memory, axis = 0))
Reward_plot.set_xlabel('Episodes')
Reward_plot.set_ylabel('Average reward ({} Runs)'.format(RUNS))

plt.figure(1)
Loss_plot = plt.subplot()
Loss_plot.plot(np.mean(loss_memory, axis = 0))
Loss_plot.set_xlabel('Episodes')
Loss_plot.set_ylabel('Average Loss ({} Runs)'.format(RUNS))

store = True
if store:
    filename = 'PG_reinforce_reward'
    outfile = open(filename,'wb')
    pickle.dump(reward_memory,outfile)
    outfile.close()

    filename = 'PG_reinforce_loss'
    outfile = open(filename,'wb')
    pickle.dump(loss_memory,outfile)
    outfile.close()

loadplot = False
if loadplot:
    file_list = ['PG_reinforce_loss','PG_reinforce_reward']
    for i,filename in enumerate(file_list):
        openfile = open(filename,'rb')
        variable = pickle.load(openfile)
        plt.figure(i)
        plt.plot(variable[0])
