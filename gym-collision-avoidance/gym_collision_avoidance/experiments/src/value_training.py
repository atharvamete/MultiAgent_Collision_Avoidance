import os
from cmath import nan
from turtle import st

import gym
import numpy as np
import pickle
import random
from operator import itemgetter

from gym_collision_avoidance.experiments.src.get_value_traj import GET_VALUE_TRAJ

gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Example'
import tensorflow as tf
import tensorflow as tf
from tensorflow import keras

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.Session().__enter__()
from helper import get_feasible_actions, get_fil_vel, get_reward, get_val, normalize_angle, propogate, to_ego, to_vec_vel

def get_random_start_goal():
    start_1_x_1 = np.random.randint(-7,-4)
    start_1_x_2 = np.random.randint(4,7)
    start_1_x = np.random.choice([start_1_x_1,start_1_x_2])
    start_1_y_1 = np.random.randint(-7,-4)
    start_1_y_2 = np.random.randint(4,7)
    start_1_y = np.random.choice([start_1_y_1,start_1_y_2])
    if start_1_x<0:
        start_2_x = np.random.randint(4,7)
        goal_1_x = np.random.randint(4,7)
        goal_2_x = np.random.randint(-7,-4)
    else:
        start_2_x = np.random.randint(-7,-4)
        goal_1_x = np.random.randint(-7,-4)
        goal_2_x = np.random.randint(4,7)
    if start_1_y<0:
        start_2_y = np.random.randint(-7,-4)
        goal_1_y = np.random.randint(4,7)
        goal_2_y = np.random.randint(4,7)
    else:
        start_2_y = np.random.randint(4,7)
        goal_1_y = np.random.randint(-7,-4)
        goal_2_y = np.random.randint(-7,-4)
    start=[[start_1_x,start_1_y],[start_2_x,start_2_y]]
    goal=[[goal_1_x,goal_1_y],[goal_2_x,goal_2_y]]
    return start, goal

def get_experience_set():
    with open('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/trajectory_dataset/2_agents/trajs/RVO5.pkl', 'rb') as f:
        data = pickle.load(f)
    gamma = 0.98
    traj_arr = []
    for traj in data:
        if traj != []:
            traj = np.array(traj)
            traj_arr.append(traj[:,:16]) #Comment out when using NORMALIZED
    traj_data = traj_arr[0]
    for i in range(len(traj_arr)):
        traj_data = np.vstack((traj_data,traj_arr[i]))
    traj_data[:,15] = gamma**(traj_data[:,15]*traj_data[:,15])
    return traj_data

if __name__ == '__main__':
    e_num = 1000
    num_episodes = 10
    m_times = 4
    experience_set = get_experience_set()
    val_network = tf.keras.models.load_model('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/weights/initialize')
    val_network_1= keras.models.clone_model(val_network)
    val_network_1.build((15,)) # replace 10 with number of variables in input layer
    val_network_1.compile(loss="mean_squared_error", optimizer="adam")
    val_network_1.set_weights(val_network.get_weights())
    env = gym.make("CollisionAvoidance-v0")
    ep = 0.5
    for episode in range(num_episodes):
        for m in range(m_times):
            start_pos,goal_pos = get_random_start_goal()#####WRITE THIS FUNCTION
            get_trajectory = GET_VALUE_TRAJ(start_pos,goal_pos,ep,val_network,val_network_1)
            print(np.shape(get_trajectory.state_trajectory))
            experience_set = np.vstack((experience_set, get_trajectory.state_trajectory))
        exp_size = np.shape(experience_set)[0]
        train_indices = random.sample(range(1, exp_size), e_num)
        train_data = experience_set[train_indices]
        val_network.fit(train_data[:,:15], train_data[:,15], batch_size=100, epochs=5)#####SORT THIS OUT USING INIT_NETWORK FILE
        if episode%3==0:
            ep = ep/2
            val_network_1= keras.models.clone_model(val_network)
            val_network_1.build((15,)) # replace 10 with number of variables in input layer
            val_network_1.compile(loss="mean_squared_error", optimizer="adam")
            val_network_1.set_weights(val_network.get_weights())
            print("af %s")
            val_network.save('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/weights/{}_epis_4'.format(episode))
        val_network.save('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/weights/10_4_{}'.format(episode))
