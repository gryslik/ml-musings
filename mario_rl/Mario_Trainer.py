import tensorflow as tf
import time
import pandas as pd
from nes_py.wrappers import JoypadSpace
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import warnings
import sys
from constants import model_path, log_fn
from helper_file import *
from DDQN_model import *
import os
import re

def train_models(previous_model=None, previous_epsilon=None):
    warnings.simplefilter("ignore", lineno=148)

    #######################################################################################
    # Initialize environment and parameters
    #######################################################################################
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)
    raw_image_dim = pre_process_image(env.reset()).shape
    num_episodes = 10000
    num_frames_to_stack = 4
    starting_episode = 0

    # if there is a model name then get file string (for loading specific model) and the old epsilon value from the log. You can then restart training.
    if (previous_model != None and previous_epsilon!=None):
        model_number = previous_model
        starting_episode = int(previous_model)
        regex = re.compile('episode-'+str(model_number)+"_", re.UNICODE)
        model_file_path = getModelPath(regex)

        old_epsilon_value = float(previous_epsilon)
        my_agent = DQN(env=env, single_frame_dim=raw_image_dim, num_frames_to_stack=num_frames_to_stack, old_model_filepath=model_file_path, old_epsilon_value=old_epsilon_value)
    else:
        my_agent = DQN(env=env, single_frame_dim=raw_image_dim, num_frames_to_stack=num_frames_to_stack)

    totalreward = []
    steps = []
    flag_result = []
    final_x_position = []

    ### This is the main execution loop
    print("Epsilon decay value is: " + str(my_agent.epsilon_decay))
    for episode in range(starting_episode, num_episodes):
        print("----------------------------------------------------------------------------")
        print("Episode: " + str(episode) + " started with memory buffer of size: " + str(
            my_agent.memory.size) + " and writing to index: " + str(my_agent.memory.index) + " and with epsilon: " + str(
            my_agent.epsilon))
        time_start = time.time()
        cur_state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2)  # reshape it (90x128x4)

        episode_reward = 0
        step = 0
        done = False
        current_x_position = []

        while not done:
            if (step % 100 == 0):
                print("At step: " + str(step))
                print_timings = True
            else:
                print_timings = False

            action = my_agent.act(cur_state)  # make a prediction
            my_agent.update_target_model(episode, step, False, False)

            new_state, reward, done, info = take_skip_frame_step(env, action, num_frames_to_stack)  # take a step when you repeat the same action for 4 frames
            step += 1

            # Add to memory
            my_agent.remember(cur_state, action, reward, new_state, done)

            # fit the model
            my_agent.replay(print_time=print_timings)

            # set the current_state to the new state
            cur_state = new_state

            episode_reward += reward
            current_x_position.append(info['x_pos'])

            if info['flag_get']:
                print("Breaking due to getting flag!")
                print("Current position is:" + str(info['x_pos']))
                break
            if step > 3000:
                print("Breaking due to out of steps.")
                break

        totalreward.append(episode_reward)
        steps.append(step)
        flag_result.append(info['flag_get'])
        final_x_position.append(current_x_position[-1])

        if info['flag_get']:
            info_str = "Episode: " + str(episode) + " -- SUCCESS -- with a total reward of: " + str(
                episode_reward) + "and at position: " + str(final_x_position[-1])
            print(info_str)
            my_agent.save_model(model_path + "episode-{}_model_success.h5".format(episode))

        else:
            info_str = "Episode: " + str(episode) + " -- FAILURE -- with a total reward of: " + str(episode_reward) + " and at position: " + str(final_x_position[-1])
            print(info_str)
            if episode % 10 == 0:
                my_agent.save_model(model_path + "episode-{}_model_failure.h5".format(episode))

        time_end = time.time()
        tf.keras.backend.clear_session()

        print("Episode: " + str(int(episode)) + " completed in steps/time/avg_running_reward: " + str(
            steps[-1]) + " / " + str(int(time_end - time_start)) + " / " + str(np.array(totalreward)[-100:].mean()))
        print("----------------------------------------------------------------------------")

        ## logging info to log file
        info_str = "Episode: " + str(int(episode)) + "; steps: " + str(steps[-1]) + "; time: " + str(
            int(time_end - time_start)) + "; epsilon: " + str(my_agent.epsilon) + "; total reward: " + str(
            episode_reward) + "; final position: " + str(final_x_position[-1]) + "; avg_running_reward: " + str(
            np.array(totalreward)[-100:].mean())

        ## write to log file
        log = open(log_fn, "a+")  # append mode and create file if it doesnt exist
        log.write(info_str +
                "\n" +
                "----------------------------------------------------------------------------" +
                "\n")
        log.close()

    env.close()

    results_df = pd.DataFrame(totalreward, columns=['episode_reward'])
    results_df['steps_taken'] = steps
    results_df['flag_get'] = flag_result
    results_df['x_pos'] = final_x_position
    results_df['average_running_reward'] = results_df['episode_reward'].rolling(window=100).mean()
    results_df.to_csv(model_path + "training_results.csv")

def run_all_models():
    warnings.simplefilter("ignore", lineno=148)

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, RIGHT_ONLY)

    all_file_names = os.listdir(model_path)
    all_file_numbers = sorted([convert_to_int(re.split('-|_', x)[1]) for x in os.listdir(model_path) if convert_to_int(re.split('-|_', x)[1]) is not None])
    sorted_file_names = [None]*len(all_file_numbers)

    for idx, model_number in enumerate(all_file_numbers):
        regex = re.compile('episode-' + str(model_number) + '_', re.UNICODE)
        sorted_file_names[idx] = list(filter(regex.match,all_file_names))[0]


    if "travel_distance.csv" in sorted_file_names:
        models_processed = pd.read_csv(model_path+"travel_distance.csv")['model_name'].values
        models_to_compute = [x for x in sorted_file_names if (x not in models_processed and x not in ".DS_Store" and x not in "travel_distance.csv")]
    else:
        models_to_compute = [item for item in sorted_file_names if (item not in "travel_distance.csv" and item not in ".DS_Store")]


    num_frames_to_stack = 4
    num_frames_to_collapse = 4
    for idx, model_name in enumerate(models_to_compute):
        print("Processing index/name: " + str(idx) + " --- " + model_name)
        try:
            time_start = time.time()
            model = tf.keras.models.load_model(model_path + model_name)

            state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2) #reshape it (120x128x4).
            done = False
            step_counter = 0
            while not done and step_counter < 500: # Now we need to take the same action every 4 steps
                prediction_values = model.predict(np.expand_dims(state, axis=0).astype('float16'))
                action = np.argmax(prediction_values)
                state, reward, done, info = take_skip_frame_step(env, action, num_frames_to_collapse, False)
                step_counter+=1

            all_files = os.listdir(model_path)
            if "travel_distance.csv" not in all_files:
                with open(model_path + "travel_distance.csv", 'a') as f:
                    pd.DataFrame([[model_name, info['x_pos']]], columns=['model_name', 'travel_distance']).to_csv(model_path + "travel_distance.csv", header=True, mode='w')
            else:
                with open(model_path + "travel_distance.csv", 'a') as f:
                    pd.DataFrame([[model_name, info['x_pos']]], columns=['model_name', 'travel_distance']).to_csv(model_path+"travel_distance.csv", header=False, mode='a')

            time_end = time.time()
            print("This took: " + str(int(time_end - time_start)) + " seconds. Ended at position: " + str(info['x_pos']))
            print('-----------------------------------------------------------------------------------------------------------')
        except:
            print("Episode: " + str(idx) + " failed!")
    env.close()

def record_video(model_number):
    save_individual_frames = False

    warnings.simplefilter("ignore", lineno=148)

    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')   #2-2, 1-1
    env = JoypadSpace(env, RIGHT_ONLY)
    env = gym.wrappers.RecordVideo(env, video_path+str(model_number))

    regex = re.compile('episode-'+str(model_number)+"_", re.UNICODE)

    model_file_path = getModelPath(regex)

    print("Processing model: " + model_file_path)
    model = tf.keras.models.load_model(model_file_path)


    num_frames_to_stack = 4
    state = np.repeat(pre_process_image(env.reset())[:, :, np.newaxis], num_frames_to_stack, axis=2) #reshape it (120x128x4).
    #show_bitmap_image(long_state[:,:,3]) sample code
    done = False
    step_counter = 0
    while not done and step_counter < 2000: # Now we need to take the same action every 4 steps
        prediction_values = model.predict(np.expand_dims(state, axis=0).astype('float16'))
        action = np.argmax(prediction_values)
        state, reward, done, info = take_skip_frame_step(env, action, 4, True)
        if save_individual_frames:
            for frame_idx in range(num_frames_to_stack):
                save_bitmap_data(state[:,:,frame_idx], video_path + str(model_number)+"/individual frames/" + str(step_counter),frame_idx)
        step_counter+=1
        print("Steps: " +  str(step_counter) + " --- position: " + str(info['x_pos']))

    env.close()