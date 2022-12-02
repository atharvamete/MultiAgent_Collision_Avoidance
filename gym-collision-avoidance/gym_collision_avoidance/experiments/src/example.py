import os
import numpy as np
import gym
gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Example'
from gym_collision_avoidance.envs import test_cases as tc
from gym_collision_avoidance.envs import Config

def main(start_pos,goal_pos):
    '''
    Minimum working example:
    2 agents: 1 running external policy, 1 running GA3C-CADRL
    '''

    # Create single tf session for all experiments
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/example/')
    # start =start_pos
    # goal = goal_pos
    # Set agent configuration (start/goal pos, radius, size, policy)
    agents = tc.get_testcase_two_agents(start_pos,goal_pos,policies=['learning', 'CADRL'])

    [agent.policy.initialize_network() for agent in agents if hasattr(agent.policy, 'initialize_network')]
    env.set_agents(agents)

    obs = env.reset() # Get agents' initial observations
    print(agents[0].vel_global_frame)
    # Repeatedly send actions to the environment based on agents' observations
    num_steps = 15
    for i in range(num_steps):

        # Query the external agents' policies
        # e.g., actions[0] = external_policy(dict_obs[0])
        actions = {}
        theta = 20*3.14/180
        theta_clip = (theta+np.pi)/(2*np.pi)
        actions[0] = np.array([0.8, theta_clip])

        # Internal agents (running a pre-learned policy defined in envs/policies)
        # will automatically query their policy during env.step
        # ==> no need to supply actions for internal agents here

        # Run a simulation step (check for collisions, move sim agents)
        obs, rewards, game_over, which_agents_done = env.step(actions)
        # print(np.array(agents[1].global_state_history[i]))
        print(rewards)

        if game_over:
            print("All agents finished!")
            break
    env.reset()

    return True

if __name__ == '__main__':
    main([[0,1],[5,-1]],[[5,3],[0,3]])
    print("Experiment over.")
