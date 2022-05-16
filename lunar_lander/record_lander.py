import tensorflow as tf
import gym
import constants
import sys
import os
import eagle_large

def record_model(file_path):
    model = tf.keras.models.load_model(file_path)

    file_name = os.path.basename(file_path)

    env = gym.make('LunarLander-v2')
    env = gym.wrappers.RecordVideo(env, constants.base_path + "/models_large_videos/" + file_name + "/")

    eagle_large.fly_lander(env, model)


record_model(sys.argv[1])