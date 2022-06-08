---
layout: post
title:  "Beating Lunar Lander with AI - Introduction"
date:   2022-05-20 14:16:40 -0400
categories: ml
---
<h3>Introduction</h3>
In reinforcement learning, an agent (an entity that makes decisions) gets rewarded or 
punished for its actions, and over time optimizes itself to better perform as it learns what actions yield the highest reward.

In this series, we use reinforcement learning to make an intelligent lunar lander playing AI.

The rules of Lunar Lander are simple. The user controls a lander by firing a combination of three thrusters:
down, left, and right. The player must fire these thrusters in a way to safely land within the area marked by the flags.

![Lunar Lander Game Guide](/images/lunar_lander.png)


<h3>Environment</h3>
To play this game with an AI, we will be using the [Gym Box2D lunar lander environment](https://www.gymlibrary.ml/environments/box2d/lunar_lander/). 
This environment takes care of many things for us, such as creating the actual game, its rules, and providing a way
for our AI to play the game. The environment specifies the following rewards and punishments:

Rewards:
* Moving from the top of the screen to the landing pad yields 100-140 points, which are lost if the lander moves away from the landing zone
* Each leg that contacts the ground yields 10 points
* If the lander comes to rest, it gets 100 points

Punishments:
* Firing the down thruster is -0.3 points per frame
* Firing either of the side thrusters is -0.03 points per frame
* Crashing the lander is -100 points

Each iteration of the game is an "episode", and an episode ends when the lander comes to rest, crashes, leaves the screen, or achieves a certain score (200).
If the agent ends the episode with a score of 200 or more, the game has been won.

An episode is split into "frames", a frame being the smallest time slice in a game (FPS in a video game). Our agent
makes calculations every frame. You'll see this referenced often.

For more information, visit the Lunar Lander environment link above.

Now that we understand our constraints, let's get our solution setup and working.

<h3>Setting up the project</h3>
First, the following must be installed for our project to run:
* [Python 3](https://www.python.org/downloads/)
* [Numpy](https://numpy.org/install/)
* [Tensorflow](https://www.tensorflow.org/install/)
* [Gym](https://pypi.org/project/gym/)
* [Pandas](https://pypi.org/project/pandas/)

Once you have the above, clone the Lunar Lander Git repo.

<h3>Running the trainer</h3>
To run the project, go into the project directory and from the terminal run:
<p><code>python3 main.py</code></p> 

When prompted, enter "1" to start training the agent. This will take a while.

<h3>While the agent is training, check
out the next blog post which talks about the project structure and what's happening as you're waiting.</h3>





