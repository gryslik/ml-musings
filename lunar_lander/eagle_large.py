import random
import numpy as np
import tensorflow as tf
import collections


# The agent we will be training
class DQN:
    def __init__(self, env):
        self.env = env
        self.memory = collections.deque(maxlen=1000000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.996
        self.learning_rate = 0.001
        self.model = self.create_model() # Will do the actual predictions

    def create_model(self):
        model = tf.keras.Sequential()
        state_shape = self.env.observation_space.shape #this returns (2,) as we observe a vector of 2 numbers (position and velocity)
        model.add(tf.keras.layers.Dense(150, input_dim = state_shape[0], activation = "relu"))
        model.add(tf.keras.layers.Dense(50, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        samples = random.sample(self.memory, batch_size)

        all_states = np.reshape([np.squeeze(x[0]) for x in samples], (batch_size, 8))
        all_actions = np.reshape([x[1] for x in samples], (batch_size, ))
        all_rewards = np.reshape([x[2] for x in samples], (batch_size, ))
        all_new_states = np.reshape([np.squeeze(x[3]) for x in samples], (batch_size, 8))
        all_dones = np.reshape([x[4] for x in samples], (batch_size, ))

        # Thus our target value is:
        future_discounted_rewards = np.array(self.model.predict_on_batch(all_new_states))  # A guess at the future discounted reward (the Q future table)
        future_max_reward = np.amax(future_discounted_rewards, axis=1)  # Figure out which reward we want -- pick the bigger one
        updated_future_discounted_rewards = all_rewards + self.gamma * future_max_reward * (~all_dones)

        all_targets = np.array(self.model.predict_on_batch(all_states))  # get us the predicted rewards
        all_targets[np.arange(len(all_targets)), np.array(all_actions)] = updated_future_discounted_rewards  # updated predicted rewards with future rewards

        # all states will predict the Q-value from our network. All targets is the q_value after taking the actiom
        # We want to make the Q value from our network prediction match our target
        self.model.train_on_batch(all_states, all_targets)

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state)[0])

    def save_model(self, fn):
        self.model.save(fn)
