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
from helper import get_feasible_actions, get_fil_vel, get_reward, get_val, normalize_angle, propogate, to_ego, to_vec_vel


class VAL_TO_TRAJ:
    def __init__(self,start_pos,goal_pos):
        self.main(start_pos,goal_pos)

    def main(self,start_pos,goal_pos):
        '''
        This file takes input as weights of the network and outputs the trajectory
        of agent got by following epsilon greedy policy.
        '''
        # Instantiate the environment
        env = gym.make("CollisionAvoidance-v0")
        # In case you want to save plots, choose the directory
        env.set_plot_save_dir(
            os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/10_4_results')

        # Set agent configuration (start/goal pos, radius, size, policy)
        agents = tc.get_testcase_two_agents(start_pos,goal_pos,policies=['RVO', 'noncoop'])
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
        print(current_state)
        state_trajectory=[to_ego(current_state),get_val(np.array([to_ego(current_state)]))]
        # Repeatedly send actions to the environment based on agents' observations
        game_over = False
        while not game_over:
            
            vel_history_x.pop(0)
            vel_history_y.pop(0)
            vel_history_x.append(current_state[6][0])
            vel_history_y.append(current_state[6][1])
            
            vel_filter = get_fil_vel(vel_history_x,vel_history_y)
            # print(current_state[6][0])
            
            sample_actions = get_feasible_actions(current_state)
            val = []
            for action in sample_actions:
                agent_vel_cmd = to_vec_vel(current_state,action[0],action[1])
                future_states = propogate(current_state, agent_vel_cmd,vel_filter,num_future_states)
                reward = get_reward(future_states)
                gamma_bar = gamma**(t_hor*current_state[4])
                value = reward + gamma_bar*get_val(np.array([to_ego(future_states[-1])]))
                val.append(value)
            val = np.array(val)
            selected_action_index = np.argmax(val)
            selected_action = sample_actions[selected_action_index]
            print(selected_action)
            actions = {}
            actions[0] = np.array([selected_action[0], normalize_angle(selected_action[1])])
            print(actions[0])
            _, _, game_over, _ = env.step(actions)

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

            state_trajectory=[to_ego(current_state),get_val(np.array([to_ego(current_state)]))]
            
            if game_over:
                print("All agents finished!")
                break
            k+=1
        env.reset()

        return state_trajectory

if __name__ == '__main__':
    start = [[0,0],[5,0]]
    goal = [[5,3],[0,3]]
    state_trajectory = VAL_TO_TRAJ(start,goal)
    print("Experiment over.")
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