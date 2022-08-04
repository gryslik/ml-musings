import numpy as np
import os
import cv2
from constants import *


def remove_color(bitmap):
    return np.mean(bitmap, axis=2).astype(np.uint8)

# downsampling code adapted from https://blog.paperspace.com/building-double-deep-q-network-super-mario-bros/
def down_sample(bitmap):
    resized_bitmap = cv2.resize(bitmap, (128, 120), interpolation=cv2.INTER_AREA)
    #focused_bitmap = resized_bitmap[18:102, :]
    #focused_bitmap = np.reshape(focused_bitmap, [84, 84])
    #return focused_bitmap
    return resized_bitmap[20:110, :]

def max_pool(bitmap):
    return np.max(bitmap, axis=0)

def getModelPath(regex):
    for file in os.listdir(model_path):
        if regex.match(file):
            return model_path+file

def pre_process_image(bitmap):
    return(down_sample(remove_color(bitmap)))

def pre_process_images(bitmaps):
    image_stack = []
    for i in range(len(bitmaps)):
        image_stack.append(pre_process_image(bitmaps[i]))
    return np.dstack(image_stack)

def take_skip_frame_step(env, action, num_frames_to_collapse, render=False):
    image_list = []
    image_reward = 0
    for i in range(num_frames_to_collapse):
        new_state, reward, done, info = env.step(action)
        if render:
            env.render()
        image_list.append(new_state.copy())
        image_reward += reward
        if done:
            num_frames_to_repeat = num_frames_to_collapse-i-1
            ## if we fail during the frames to collapse, we will simply append the last frame enough times to preserve shape
            for i in range(num_frames_to_repeat):
                image_list.append(new_state.copy())
                image_reward += reward
            break
    combined_state = pre_process_images(image_list)
    return combined_state, image_reward, done, info

def convert_to_int(value):
    try:
        return(int(value))
    except Exception as e:
        return None

def save_bitmap_data(data,fpath, frame_idx):
    from matplotlib import pyplot as plt
    os.makedirs(fpath, exist_ok=True)
    plt.imsave(fname=fpath + "/" + str(frame_idx) + ".png", arr=data, cmap="gray")

def show_bitmap_data(data):
    from matplotlib import pyplot as plt
    plt.imshow(X=data, cmap="gray")
    plt.show()

def training_reward_analysis():
    import matplotlib.pyplot as plt
    file1 = open(log_path, 'r')
    Lines = file1.readlines()

    reward_data = np.empty(0)

    for line in Lines:
        if line == '----------------------------------------------------------------------------\n':
            pass
        else:
            data = line.split(" ")
            reward = float(data[len(data)-1][:-1])
            reward_data = np.append(reward_data, reward)

    plt.scatter(range(reward_data.shape[0]), reward_data)
    