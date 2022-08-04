---
layout: post
title:  "Part 2"
date:   2022-05-21 14:16:40 -0400
author: "Daniel Mogilevsky"
categories: Lander
---
<h3>Introduction</h3>
In the previous post, we provided an overview of the lunar lander environment and got our solution
installed and running. In this post, we'll talk about the structure of the solution and what's happening as we train
our agent.

<h3>Structure</h3>
By now you've probably taken note of the files in the project directory:
* Constants.py: Stores directory paths and strings we use throughout the project
* main.py: Starting point for the project that calls the necessary functions based on user inputs
* eagle_large.py: File with all the AI logic and functions for training, recording videos, saving models, etc

Let's dive into how the learning agent was created
<h3>Creating the learning agent</h3>
Our agent is responsible for fine tuning the model used to fly the lander, below are the parts of the agent
we had to define before the training process could start.

<h4>Hyper Parameters</h4>
Hyper parameters are parameters that control the learning process rather than the performance of the model itself. 
The following hyper parameters must be decided when creating our learning agent:

* Learning rate (from 0-1): This determines how quickly the agent will pick up new values
* Gamma a.k.a Discount factor (0-1): How much weight is given to the reward of future actions when calculating the value
of a certain action
* Epsilon decay, initial, and minimum: Epsilon is the probability that we choose a random action for the current frame instead of a 
optimal action. Epsilon decay is the value by which epsilon decreases each frame.
* Memory store: How large of a memory store we want to keep. In our agents memory, we store a state, action, reward for the
state+action, the resulting state, and whether this action ended the episode.

Through researching how other people have solved this environment the following hyper parameters were discovered to be optimal:
[Code Link](https://github.com/gryslik/ml-musings/blob/109d54e39476636e714b686a1c63ef71da54d1ae/lunar_lander/eagle_large.py#L16-L25)
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
In a later blog post, we'll discuss why these particular parameters are optimal. For now, it's sufficient to understand
what they are and how they impact our agent.

<h4>Model Structure - Neural Net</h4>
The agent itself doesn't fly the lander, and the hyper parameters just inform how the agent trains itself, the _model_
is what actually flies the lander. The model is a part of the agent and something the agent fine-tunes over time. 
We are using a neural net for our model. A neural net consists of an input, hidden, and output layer
and takes input (the agent's observations about the environment) and turns them into outputs (actions).
We'll discuss how exactly a neural net works later on, for now it is sufficient to understand the above. 
For our neural net, we are using a [Keras Sequential Model](https://keras.io/guides/sequential_model/)

We have found the following structure to work well:
* Input layer with 8 nodes representing the observation space ((x and y coordinates of the lander, x and y linear velocities, angle, angular velocity, two booleans representing the legs
  touching the ground))
* Hidden layer with 150 nodes
* Hidden layer with 120 nodes
* Output layer with 4 nodes representing the action space (do nothing, fire left engine, fire main engine, fire right engine)

Below is the code for this initialization
[Code Link](https://github.com/gryslik/ml-musings/blob/109d54e39476636e714b686a1c63ef71da54d1ae/lunar_lander/eagle_large.py#L27-L34)
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

Once again, don't sweat the details right now, in our next post, we'll do that in the final post.

Lastly, our class has two helper functions, _act_ and _remember_

_Act_, shown below, is causes the agent to take an action given a game/environment state. The details of how epsilon
is used will be made clear in the last post, but astute readers will see that epsilon determines whether the
agent picks a random action or what it believes to be the ideal action.
[Code Link](https://github.com/gryslik/ml-musings/blob/109d54e39476636e714b686a1c63ef71da54d1ae/lunar_lander/eagle_large.py#L63-L69)
```python
    def act(self, state):
        self.epsilon *= self.epsilon_decay # Multiply our epsilon by the decay
        self.epsilon = max(self.epsilon_min, self.epsilon) # Never let epsilon go below the minium value
        if np.random.random() < self.epsilon: # Generate a random number 0-1, if it's less than episolon, do a random action
            return self.env.action_space.sample()
        else: # Otherwise, pick what we believe to be the best action
            return np.argmax(self.model.predict(state)[0])
```

_Remember_ causes the agent to remember the previous state, the action it took in that state, the reward of the action,
the resulting/new state from that action, and whether the game reached completion from the action. It remembers this
by appending all of these details into the memory buffer
[Code Link](https://github.com/gryslik/ml-musings/blob/109d54e39476636e714b686a1c63ef71da54d1ae/lunar_lander/eagle_large.py#L37)
```python
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
```

<h3>Training the agent</h3>

If you followed the instructions on the previous blog post and have the training running in the background,
you've likely taken note of the output and perhaps even saw an instance or two of the lander in action.

Let's dive into what's happening in the training cycle.

We started by creating the environment, picking the number of episodes we'll train the agent for, initializing
the agent, and creating variables to track total reward and # of steps per episode

[Link to Code](https://github.com/gryslik/ml-musings/blob/109d54e39476636e714b686a1c63ef71da54d1ae/lunar_lander/eagle_large.py#L155-L196)
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
        cur_state = env.reset().reshape(1,8) # Reset the environment
        episode_reward = 0
        step = 0
```

Until the episode terminates, the following cycle occurs:
```python
while True:
```
1. The agent decides on an action based on the current state.
```python
action = my_agent.act(cur_state)
```
2. We take the action and get back a new state, the reward for the action, and whether we are done with the episode(as per the episode
   termination conditions mentioned before).
```python
new_state, reward, done, info = env.step(action)
new_state = new_state.reshape(1, 8)
```
3. We make the agent remember the action it took on the previous state and
   the result of that action
```python
my_agent.remember(cur_state, action, reward, new_state, done)
```
4. The agent replays this event to learn from it.
```python
my_agent.replay()
```
5. We update environment state, add the reward of the action taken to the total reward, and increment the step.
   We also check if we are done with this episode.
```python
            cur_state = new_state
            episode_reward += reward
            step +=1
            if done:
                break
```

These values are what you see printed to the console when you train the agent. 

Here's how the models progressed as we trained them:

First attempt (model 0) goes terribly
![crash](/videos/lander_crash.gif)


By model 40, the lander learns how to hover above the goal area without crashing:
![hover](/videos/lander_hover.gif)

At model 110, the lander is landing, but not quite in the goal post
![almost](/videos/almost.gif)

By model/episode 140, the lander is landing in the goal area perfectly:
![perfect](/videos/perfect.gif)

<h3>Conclusion</h3>

After only 140 episodes of training, our AI can play lunar lander better than
most human players. In our final blog post, we'll analyze why our hyper parameters and neural net structure work
and the math behind them.