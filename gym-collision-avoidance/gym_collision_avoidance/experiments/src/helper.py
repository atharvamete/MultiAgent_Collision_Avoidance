import tensorflow as tf
import numpy as np
val_network = tf.keras.models.load_model('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/weights/10_4')
dt = 0.1

def clip_angle(angle):
    if angle>np.pi:
        angle-=2*np.pi
    if angle<-np.pi:
        angle+=2*np.pi
    return angle

def normalize_angle(angle):
    return (angle+np.pi)/(2*np.pi)

def get_feasible_actions(state):
    sample_actions=[]
    for v in np.arange(0.1,state[4],0.2):
        for theta in np.arange(-30*3.14/180,30*3.14/180,10*3.14/180):
            sample_actions.append([v,theta])
    return sample_actions

def to_vec_vel(state,vel,theta):
    head = clip_angle(state[2]+theta)
    vx = vel*np.cos(head)
    vy = vel*np.sin(head)
    return vx, vy

def to_ego(state):
    goal_global_frame = state[8]
    pos_global_frame = state[0]
    heading_global_frame = state[2]
    other_agent_pos_global_frame = state[5]
    goal_direction = goal_global_frame - pos_global_frame 
    theta = np.arctan2(goal_direction[1], goal_direction[0])
    T_global_ego = np.array([[np.cos(theta), -np.sin(theta), pos_global_frame[0]], [np.sin(theta), np.cos(theta), pos_global_frame[1]], [0,0,1]])
    T_global_ego_vel = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    other_agent_ego_pos = np.dot(np.linalg.inv(T_global_ego), np.array([other_agent_pos_global_frame[0], other_agent_pos_global_frame[1], 1]))
    dg = np.linalg.norm(pos_global_frame - goal_global_frame)
    da = np.linalg.norm(pos_global_frame - other_agent_pos_global_frame)
    theta_ego = clip_angle(theta-heading_global_frame)
    v_a = np.dot(np.linalg.inv(T_global_ego_vel), state[1])
    v_oa = np.dot(np.linalg.inv(T_global_ego_vel), state[6])
    r_a = state[3]
    r_oa = state[7]
    vpref = state[4]
    joint_state = [dg,vpref,v_a[0],v_a[1],r_a,theta_ego,v_oa[0],v_oa[1],other_agent_ego_pos[0],other_agent_ego_pos[1],r_oa,r_a+r_oa,np.cos(theta_ego),np.sin(theta_ego),da]
    return joint_state

def get_val(joint_state):
    return val_network.predict(joint_state)

def get_fil_vel(velx_list,vely_list):
    vx = max(0, min(np.nanmean(np.array(velx_list)),1))
    vy = max(0, min(np.nanmean(np.array(vely_list)),1))
    return [vx,vy]

def propogate(state,a_vel,oa_vel,num_future_states):
    future_states = []
    new_state = np.copy(state)
    for i in range(num_future_states):     
        new_state[0] = new_state[0]+dt*np.array(a_vel)
        new_state[1] = a_vel
        new_state[2] = np.arctan2(a_vel[1],a_vel[0])
        new_state[5] = new_state[5]+dt*np.array(oa_vel)
        new_state[6] = oa_vel
        new_state = np.copy(new_state)
        future_states.append(new_state)
    return future_states

def get_reward(future_states):
    # future_states = propogate(state,a_vel,oa_vel,num_future_states)
    dba = []
    dtg = []
    for states in future_states:
        dba.append(np.linalg.norm(states[0]-states[5])-(states[3]+states[7]))
        dtg.append(np.linalg.norm(states[0]-states[8])-states[3])
    dmin = np.min(dba)
    dtg = np.array(dtg)
    if dmin<0.0:
        return -0.25
    elif dmin<0.2:
        return (-0.1-dmin/2)
    elif dtg.any()<0.05:
        return 1.0
    else:
        return 0.0
