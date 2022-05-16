import numpy as np
import tensorflow as tf
import gym
import time
import pandas as pd
import constants
from eagle_large import DQN

#Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector.
# Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
# If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or
# comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main
# engine is -0.3 points each frame. Firing side engine is -0.03 points each frame. Solved is 200 points.
#

#### State informaton
# X- position
# Y- position
# X-velocity
# Y-velocity
#lander angle
#angular velocity
#left leg ground
#right light ground

def train_agent():
    env = gym.make('LunarLander-v2')
    num_episodes = 500
    my_agent = DQN(env=env)
    totalreward = []
    steps = []
    for episode in range(num_episodes):
        print("======================================================")
        print("Processing episode: " + str(episode))
        print("======================================================")
        time_start = time.time()
        cur_state = env.reset().reshape(1,8)
        episode_reward = 0
        step = 0
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
        print("--------------------------------------------------------")
        print("Episode: " + str(int(episode)) + " completed in: " + str(step) + " steps.")
        print("--------------------------------------------------------")
        if episode_reward < 200.0:
            print("Failed to complete episode: " + str(episode) + " with a total reward of: " + str(episode_reward))
            if episode % 10 == 0:
                my_agent.save_model(constants.fail_path + constants.model_name + "-episode-{}_model_failure.h5".format(episode))
        else:
            print("Successfully completed in episode: " + str(episode) + " with a total reward of: " + str(episode_reward))
            my_agent.save_model(constants.success_path + constants.model_name + "-episode-{}_model_success.h5".format(episode))
        time_end = time.time()
        tf.keras.backend.clear_session()
        print("Processing episode: " + str(episode) + " took: " + str(int(time_end - time_start)) + " seconds. Avg running reward is: " + str(np.array(totalreward)[-100:].mean()))
    env.close()

    results_df = pd.DataFrame(totalreward, columns = ['episode_reward'])
    results_df['steps_taken'] = steps
    results_df['Success'] = results_df['episode_reward'] > 200.0
    results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=100).mean()

    results_df.to_csv(constants.model_path+"-training_results.csv")

train_agent()