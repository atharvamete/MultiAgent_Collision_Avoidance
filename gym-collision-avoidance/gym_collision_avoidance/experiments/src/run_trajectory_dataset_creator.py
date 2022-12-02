import os
import numpy as np
import pickle
from tqdm import tqdm

from gym_collision_avoidance.envs.config import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env

from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy

np.random.seed(0)

Config.EVALUATE_MODE = True
Config.SAVE_EPISODE_PLOTS = False
Config.SHOW_EPISODE_PLOTS = False
Config.DT = 0.1
start_from_last_configuration = False

results_subdir = 'trajectory_dataset'

# test_case_fn = tc.get_testcase_2agents_swap
test_case_fn = tc.get_testcase_random
policies = {
            'RVO': {
                'policy': RVOPolicy,
                },
            # 'GA3C-CADRL-10': {
            #     'policy': GA3CCADRLPolicy,
            #     'checkpt_dir': 'IROS18',
            #     'checkpt_name': 'network_01900000'
            #     },
            }

num_agents_to_test = [2]
num_test_cases = 500
test_case_args = {}
Config.PLOT_CIRCLES_ALONG_TRAJ = True
Config.NUM_TEST_CASES = num_test_cases

def clip_angle(angle):
    if angle>np.pi:
        angle-=2*np.pi
    if angle<-np.pi:
        angle+=2*np.pi
    return angle

def clip_angle_pos(angle):
    if angle<0:
        angle+=2*np.pi
    return angle

def add_traj(agents, trajs, dt, traj_i, max_ts):
    agent_i = 0
    other_agent_i = (agent_i + 1) % 2
    agent = agents[agent_i]
    other_agent = agents[other_agent_i]
    agent1_t = int(max_ts[agent_i])
    agent2_t = int(max_ts[other_agent_i])
    future_plan_horizon_secs = 3.0
    future_plan_horizon_steps = int(future_plan_horizon_secs / dt)
    # print(len(other_agent.global_state_history))
    for t in range(agent1_t):
        # print(agent2_t)
        ttg = (agent1_t-t-1)*dt
        if t<agent2_t:
            robot_linear_speed = agent.global_state_history[t, 9]
            robot_angular_speed = agent.global_state_history[t, 10] / dt

            t_horizon = min(agent1_t, t+future_plan_horizon_steps)
            future_linear_speeds = agent.global_state_history[t:t_horizon, 9]
            future_angular_speeds = agent.global_state_history[t:t_horizon, 10] / dt
            predicted_cmd = np.dstack([future_linear_speeds, future_angular_speeds])

            future_positions = agent.global_state_history[t:t_horizon, 1:3]

            d = {
                'control_command': np.array([
                    robot_linear_speed,
                    robot_angular_speed
                    ]),
                'predicted_cmd': predicted_cmd,
                'future_positions': future_positions,
                'other_agent_state': {
                    'position': np.array([
                        other_agent.global_state_history[t, 1],
                        other_agent.global_state_history[t, 2],
                        ]),
                    'velocity': np.array([
                        other_agent.global_state_history[t, 7],
                        other_agent.global_state_history[t, 8],
                        ])
                },
                'robot_state': np.array([
                    agent.global_state_history[t, 1],
                    agent.global_state_history[t, 2],
                    agent.global_state_history[t, 10],
                    ]),
                'goal_position': np.array([
                    agent.goal_global_frame[0],
                    agent.goal_global_frame[1],
                    ])
            }
            goal_global_frame = np.array([agent.goal_global_frame[0],agent.goal_global_frame[1]])
            pos_global_frame = np.array([agent.global_state_history[t, 1],agent.global_state_history[t, 2]])
            heading_global_frame = agent.global_state_history[t, 10]
            # print(heading_global_frame)
            other_agent_pos_global_frame = np.array([other_agent.global_state_history[t, 1],other_agent.global_state_history[t, 2]])
            goal_direction = goal_global_frame - pos_global_frame 
            theta = np.arctan2(goal_direction[1], goal_direction[0])
            # print(heading_global_frame*180/3.14,theta*180/3.14)
            T_global_ego = np.array([[np.cos(theta), -np.sin(theta), pos_global_frame[0]], [np.sin(theta), np.cos(theta), pos_global_frame[1]], [0,0,1]])
            T_global_ego_vel = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            other_agent_ego_pos = np.dot(np.linalg.inv(T_global_ego), np.array([other_agent_pos_global_frame[0], other_agent_pos_global_frame[1], 1]))
            # print(other_agent_ego_pos,"other agent ego")
            dg = np.linalg.norm(pos_global_frame - goal_global_frame)
            da = np.linalg.norm(pos_global_frame - other_agent_pos_global_frame)
            theta_ego = clip_angle(theta-heading_global_frame)
            v_a = np.dot(np.linalg.inv(T_global_ego_vel), np.array([agent.global_state_history[t, 7], agent.global_state_history[t, 8]]))
            v_oa = np.dot(np.linalg.inv(T_global_ego_vel), np.array([other_agent.global_state_history[t, 7], other_agent.global_state_history[t, 8]]))
            # print(v_oa)
            r_a = agent.radius
            r_oa = other_agent.radius
            vpref = agent.pref_speed
            joint_state = np.array([dg,vpref,v_a[0],v_a[1],r_a,theta_ego,v_oa[0],v_oa[1],other_agent_ego_pos[0],other_agent_ego_pos[1],r_oa,r_a+r_oa,np.cos(theta_ego),np.sin(theta_ego),da,ttg])
            trajs[traj_i].append(joint_state)
        else:
            robot_linear_speed = agent.global_state_history[t, 9]
            robot_angular_speed = agent.global_state_history[t, 10] / dt

            t_horizon = min(agent1_t, t+future_plan_horizon_steps)
            future_linear_speeds = agent.global_state_history[t:t_horizon, 9]
            future_angular_speeds = agent.global_state_history[t:t_horizon, 10] / dt
            predicted_cmd = np.dstack([future_linear_speeds, future_angular_speeds])

            future_positions = agent.global_state_history[t:t_horizon, 1:3]

            d = {
                'control_command': np.array([
                    robot_linear_speed,
                    robot_angular_speed
                    ]),
                'predicted_cmd': predicted_cmd,
                'future_positions': future_positions,
                'other_agent_state': {
                    'position': np.array([
                        other_agent.global_state_history[agent2_t-1, 1],
                        other_agent.global_state_history[agent2_t-1, 2],
                        ]),
                    'velocity': np.array([
                        other_agent.global_state_history[agent2_t-1, 7],
                        other_agent.global_state_history[agent2_t-1, 8],
                        ])
                },
                'robot_state': np.array([
                    agent.global_state_history[t, 1],
                    agent.global_state_history[t, 2],
                    agent.global_state_history[t, 10],
                    ]),
                'goal_position': np.array([
                    agent.goal_global_frame[0],
                    agent.goal_global_frame[1],
                    ])
            }
            goal_global_frame = np.array([agent.goal_global_frame[0],agent.goal_global_frame[1]])
            pos_global_frame = np.array([agent.global_state_history[t, 1],agent.global_state_history[t, 2]])
            heading_global_frame = agent.global_state_history[t, 10]
            # print(heading_global_frame)
            other_agent_pos_global_frame = np.array([other_agent.global_state_history[agent2_t-1, 1],other_agent.global_state_history[agent2_t-1, 2]])
            goal_direction = goal_global_frame - pos_global_frame 
            theta = np.arctan2(goal_direction[1], goal_direction[0])
            T_global_ego = np.array([[np.cos(theta), -np.sin(theta), pos_global_frame[0]], [np.sin(theta), np.cos(theta), pos_global_frame[1]], [0,0,1]])
            T_global_ego_vel = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            other_agent_ego_pos = np.dot(np.linalg.inv(T_global_ego), np.array([other_agent_pos_global_frame[0], other_agent_pos_global_frame[1], 1]))
            # print(other_agent_ego_pos,"other agent ego")
            dg = np.linalg.norm(pos_global_frame - goal_global_frame)
            da = np.linalg.norm(pos_global_frame - other_agent_pos_global_frame)
            theta_ego = clip_angle(theta-heading_global_frame)
            v_a = np.dot(np.linalg.inv(T_global_ego_vel), np.array([agent.global_state_history[t, 7], agent.global_state_history[t, 8]]))
            v_oa = np.dot(np.linalg.inv(T_global_ego_vel), np.array([other_agent.global_state_history[agent2_t-1, 7], other_agent.global_state_history[agent2_t-1, 8]]))
            # print(v_oa,[other_agent.global_state_history[agent2_t-1, 7], other_agent.global_state_history[agent2_t-1, 8]])
            r_a = agent.radius
            r_oa = other_agent.radius
            vpref = agent.pref_speed
            joint_state = np.array([dg,vpref,v_a[0],v_a[1],r_a,theta_ego,v_oa[0],v_oa[1],other_agent_ego_pos[0],other_agent_ego_pos[1],r_oa,r_a+r_oa,np.cos(theta_ego),np.sin(theta_ego),da,ttg])
            trajs[traj_i].append(joint_state)
            # trajs[traj_i].append(d)

#     global_state = np.array([self.t,
#                                  self.pos_global_frame[0],
#                                  self.pos_global_frame[1],
#                                  self.goal_global_frame[0],
#                                  self.goal_global_frame[1],
#                                  self.radius,
#                                  self.pref_speed,
#                                  self.vel_global_frame[0],
#                                  self.vel_global_frame[1],
#                                  self.speed_global_frame,
#                                  self.heading_global_frame])

    return trajs


def main():
    env, one_env = create_env()
    dt = one_env.dt_nominal
    file_dir_template = os.path.dirname(os.path.realpath(__file__)) + '/../results/{results_subdir}/{num_agents}_agents'

    trajs = [[] for _ in range(num_test_cases)]

    for num_agents in num_agents_to_test:
        
        file_dir = file_dir_template.format(num_agents=num_agents, results_subdir=results_subdir)
        plot_save_dir = file_dir + '/figs/'
        os.makedirs(plot_save_dir, exist_ok=True)
        one_env.plot_save_dir = plot_save_dir

        test_case_args['num_agents'] = num_agents
        test_case_args['side_length'] = 7
        for test_case in tqdm(range(num_test_cases)):
            # test_case_args['test_case_index'] = test_case
            # test_case_args['num_test_cases'] = num_test_cases
            for policy in policies:
                one_env.plot_policy_name = policy
                policy_class = policies[policy]['policy']
                test_case_args['policies'] = 'RVO'
                agents = test_case_fn(**test_case_args)
                for agent in agents:
                    if 'checkpt_name' in policies[policy]:
                        agent.policy.env = env
                        agent.policy.initialize_network(**policies[policy])
                        print('inifloop')
                one_env.set_agents(agents)
                one_env.test_case_index = test_case
                init_obs = env.reset()
                episode_stats_t, agent_t = run_episode(env, one_env)
                times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck, agents = episode_stats_t["time_to_goal"], episode_stats_t["extra_time_to_goal"], episode_stats_t["collision"], episode_stats_t["all_at_goal"], episode_stats_t["any_stuck"], agent_t
                # times_to_goal, extra_times_to_goal, collision, all_at_goal, any_stuck, agents = run_episode(env, one_env)
                # print(agents)
                if all_at_goal:
                    max_ts = [t / dt for t in times_to_goal]
                    trajs = add_traj(agents, trajs, dt, test_case, max_ts)

        # print(trajs)
                
        one_env.reset()

        pkl_dir = file_dir + '/trajs/'
        os.makedirs(pkl_dir, exist_ok=True)
        fname = pkl_dir+policy+'5'+'.pkl'
        pickle.dump(trajs, open(fname,'wb'))
        print('dumped {}'.format(fname))

    print("Experiment over.")

if __name__ == '__main__':
    main()