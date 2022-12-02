import os
from cmath import nan

import gym
import numpy as np

gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Example'
import tensorflow as tf
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import test_cases as tc

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.Session().__enter__()
from helper import get_feasible_actions, get_fil_vel, get_reward, normalize_angle, propogate, to_ego, to_vec_vel


class GET_VALUE_TRAJ:
    def __init__(self,start_pos,goal_pos,ep,val_network,val_network_1):
        self.main(start_pos,goal_pos,ep,val_network,val_network_1)

    def get_max_val(self,current_state,vel_filter,num_future_states,gamma,t_hor,val_network_1):
        sample_actions = get_feasible_actions(current_state)
        val = []
        for action in sample_actions:
            agent_vel_cmd = to_vec_vel(current_state,action[0],action[1])
            future_states = propogate(current_state, agent_vel_cmd,vel_filter,num_future_states)
            reward = get_reward(future_states)
            gamma_bar = gamma**(t_hor*current_state[4])
            value = reward + gamma_bar*(val_network_1.predict(np.array([to_ego(future_states[-1])])))
            val.append(value)
        val = np.array(val)
        return np.max(val)

    def main(self,start_pos,goal_pos,ep,val_network,val_network_1):
        '''
        This file takes input as weights of the network and outputs the trajectory
        of agent got by following epsilon greedy policy.
        '''
        # Instantiate the environment
        env = gym.make("CollisionAvoidance-v0")
        # In case you want to save plots, choose the directory
        env.set_plot_save_dir(
            os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/get_traj_train/')

        # Set agent configuration (start/goal pos, radius, size, policy)
        agents = tc.get_testcase_two_agents(start_pos,goal_pos,policies=['learning', 'noncoop'])
        [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
        env.set_agents(agents)
        obs = env.reset() # Get agents' initial observations
        
        #set initial state trajectory
        pos_1 = agents[0].pos_global_frame
        vel_1 = agents[0].vel_global_frame
        head_1 = agents[0].heading_global_frame
        rad_1 = agents[0].radius
        vpref_1 = agents[0].pref_speed
        pos_2 = agents[1].pos_global_frame
        vel_2 = agents[1].vel_global_frame
        rad_2 = agents[1].radius
        goal_1 = agents[0].goal_global_frame
        current_state = np.array([pos_1,vel_1,head_1,rad_1,vpref_1,pos_2,vel_2,rad_2,goal_1])
        vel_history_x = [np.nan]*5
        vel_history_y = [np.nan]*5
        k = 1 #iteration no.
        gamma = 0.98
        t_hor = 1.0
        dt = agents[0].dt_nominal
        num_future_states = int(t_hor/dt)
        # print(current_state)
        state_trajectory= [np.array(to_ego(current_state)+[0.1])]
        vel_history_x.pop(0)
        vel_history_y.pop(0)
        vel_history_x.append(current_state[6][0])
        vel_history_y.append(current_state[6][1])
        
        vel_filter = get_fil_vel(vel_history_x,vel_history_y)
        # Repeatedly send actions to the environment based on agents' observations
        game_over = False
        while not game_over:

            sample_actions = get_feasible_actions(current_state)
            val = []
            for action in sample_actions:
                agent_vel_cmd = to_vec_vel(current_state,action[0],action[1])
                future_states = propogate(current_state, agent_vel_cmd,vel_filter,num_future_states)
                reward = get_reward(future_states)
                gamma_bar = gamma**(t_hor*current_state[4])
                value = reward + gamma_bar*val_network.predict(np.array([to_ego(future_states[-1])]))
                val.append(value)
            val = np.array(val)

            wtd = np.random.choice(range(0,2), p=[ep,1-ep])
            if wtd==0:
                selected_action_index = np.random.randint(len(sample_actions))
            else:
                selected_action_index = np.argmax(val)
            selected_action = sample_actions[selected_action_index]

            actions = {}
            actions[0] = np.array([selected_action[0], normalize_angle(selected_action[1])])
            _, ac_reward, game_over, _ = env.step(actions)

            pos_1 = agents[0].pos_global_frame
            vel_1 = agents[0].vel_global_frame
            head_1 = agents[0].heading_global_frame
            rad_1 = agents[0].radius
            vpref_1 = agents[0].pref_speed
            pos_2 = agents[1].pos_global_frame
            vel_2 = agents[1].vel_global_frame
            rad_2 = agents[1].radius
            goal_1 = agents[0].goal_global_frame
            current_state = np.array([pos_1,vel_1,head_1,rad_1,vpref_1,pos_2,vel_2,rad_2,goal_1])

            vel_history_x.pop(0)
            vel_history_y.pop(0)
            vel_history_x.append(current_state[6][0])
            vel_history_y.append(current_state[6][1])
            vel_filter = get_fil_vel(vel_history_x,vel_history_y)

            state_trajectory.append(np.array(to_ego(current_state)+[ac_reward[0]+gamma*self.get_max_val(current_state,vel_filter,num_future_states,gamma,t_hor,val_network_1)]))
            
            if game_over:
                print("All agents finished!")
                break
            k+=1
        env.reset()

        self.state_trajectory = np.array(state_trajectory)

if __name__ == '__main__':
    start = [[0,0],[5,0]]
    goal = [[5,3],[0,3]]
    ep = 0.4
    val_network = tf.keras.models.load_model('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/weights/initialize')
    val_to_train = GET_VALUE_TRAJ(start,goal,ep,val_network)
    print(np.shape(val_to_train.state_trajectory))
    '''
    pos_1 = np.array([agents[0].global_state_history[k, 1],agents[0].global_state_history[k,2]])
    vel_1 = np.array([agents[0].global_state_history[k, 7],agents[0].global_state_history[k,8]])
    head_1 = agents[0].global_state_history[k,10]
    rad_1 = agents[0].radius
    vpref_1 = agents[0].pref_speed
    pos_2 = np.array([agents[1].global_state_history[k, 1],agents[1].global_state_history[k,2]])
    vel_2 = np.array([agents[1].global_state_history[k, 7],agents[1].global_state_history[k,8]])
    rad_2 = agents[1].radius
    goal_1 = np.array([agents[0].goal_global_frame[0],agents[0].goal_global_frame[1]])
    current_state = np.array([pos_1,vel_1,head_1,rad_1,vpref_1,pos_2,vel_2,rad_2,goal_1])
    '''