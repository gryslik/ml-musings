---
layout: post
title:  "Training an agent to win at Lunar Lander"
date:   2022-05-20 14:16:40 -0400
categories: ml
---
<h3>Introduction</h3>
In reinforcement learning, an agent gets rewarded or punished for its actions in a given environment, and over
time optimizes itself to better perform as it learns what actions to take.

In this blog post, we will demonstrate reinforcement learning by training an AI to win Lunar Lander, a classic
arcade game from 1979.

The rules of the game are simple. The user controls a lander by firing a combination of three thrusters:
down, left, and right. The player must fire these thrusters in a way to safely land within the area marked by the flags.

![Lunar Lander Game Guide](/images/lunar_lander.png)


<h3>Environment</h3>
This learning environment specifies the following rewards and punishments:

Rewards:
* Moving from the top of the screen to the landing pad yields 100-140 points, which are lost if the lander moves away from the landing zone
* Each leg that contacts the ground yields 10 points
* If the lander comes to rest, it gets 100 points

Punishments:
* Firing the down thruster is -0.3 points per frame
* Firing either of the side thrusters is -0.03 points per frame
* Crashing the lander is -100 points

The game is "won" if the agent can get 200 points in a game, and a game ends when the lander comes to rest, crashes,
or leaves the screen

More information on the learning environment can be found [here](https://www.gymlibrary.ml/environments/box2d/lunar_lander/)

<h3>Creating the learning agent</h3>
The following parameters must be decided when creating our learning agent:
* Learning rate (from 0-1): This determines how quickly the agent will pick up new values
* Discount factor/gamma (0-1): How much weight is given to future rewards in the value function
* Epsilon decay, initial, and minimum: Epsilon is probability that we chose a random action instead of a greedy action, and thus
epsilon decay is the value by which epsilon decreases each episode
* Memory store: How large of a memory store we want to keep

Through research the following hyperparameters were discovered to be optimal:
```python
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
 ```

Next, we create the neural net structure for the agent. Using the Keras sequential model, we have found the following
structure to work well:

* Input layer with 150 nodes, relu activation function. The input layer is 8 dimensions corresponding to the observation space
  (x and y coordinates of the lander, x and y linear velocities, angle, angular velocity, two booleans representing the legs
touching the ground)
* Hidden layer with 120 nodes, relu activation function
* Output layer with 4 nodes, linear activation function, representing the action space (do nothing, fire left engine, fire main engine, fire right engine)

Below is the code for this initialization
```python
create_model(self):
        model = tf.keras.Sequential()
        state_shape = self.env.observation_space.shape
        model.add(tf.keras.layers.Dense(150, input_dim = state_shape[0], activation = "relu"))
        model.add(tf.keras.layers.Dense(120, activation="relu"))
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation="linear"))
        model.compile(loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model
```

<h3>Training the agent</h3>

We are now ready to train our learning agent and create effective AI models for flying the lander.

Let's start by creating the environment, picking the number of episodes we'll train the agent for, initializing
the agent, and creating variables to track total reward and # of steps per episode

```python
def train_agent():
    env = gym.make('LunarLander-v2')
    num_episodes = 500
    my_agent = DQN(env=env)
    totalreward = []
    steps = []
 ```

For each episode, we start at step 0 with a fresh environment and an accumulated reward of 0. 
```python
    for episode in range(num_episodes):
        print("======================================================")
        print("Processing episode: " + str(episode))
        print("======================================================")
        time_start = time.time()
        cur_state = env.reset().reshape(1,8)
        episode_reward = 0
        step = 0
```

Until the episode terminates, the following cycle occurs:
1. The agent decides on an action based on the current state. 
2. We take the action and get back a new state, the reward for the action, and whether we are done with the episode(as per the episode
termination conditions mentioned before).
3. We make the agent remember the action it took on the previous state and
the result of that action
4. The agent replays this event to learn from it.

```python
        while True: #will auto terminate when it reaches 200
            action = my_agent.act(cur_state)
            new_state, reward, done, info = env.step(action)
            new_state = new_state.reshape(1, 8)
            my_agent.remember(cur_state, action, reward, new_state, done)
            my_agent.replay()
            cur_state = new_state
            episode_reward += reward
            step +=1
            if done:
                break
        totalreward.append(episode_reward)
        steps.append(step)
```

Here's how this looks:

First attempt (model 0) goes terribly
![crash](/videos/lander_crash.gif)


By model 40, the lander learns how to hover above the goal area without crashing:
![hover](/videos/lander_hover.gif)

At model 110, the lander is landing, but not quite in the goal post
![almost](/videos/almost.gif)

By model/episode 140, the lander is landing in the goal area perfectly:
![perfect](/videos/perfect.gif)





