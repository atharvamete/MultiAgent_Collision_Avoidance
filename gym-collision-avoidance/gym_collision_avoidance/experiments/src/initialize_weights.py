import pickle
from matplotlib.pyplot import axis
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.utils.vis_utils import plot_model
    
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
# initial_exp = []
# for i in range(len(traj_data)):
#     initial_exp.append(np.array([np.array(traj_data[i,:15]), gamma**(traj_data[i,15]*traj_data[i,1])]))
# initial_exp = np.array(initial_exp)
# print(np.shape(traj_data)[0])
# print(np.shape(traj_data),np.shape(val_data))
# for i in traj_data:
#     if i.any()>1:
#         print(i)
# for i in range(15):
#     traj_data[:,i] = (traj_data[:,i]-np.min(traj_data[:,i]))/(np.max(traj_data[:,i])-np.min(traj_data[:,i]))
# for i in range(15):
#     print(np.min(traj_data[:,i]))
# print(np.min(val_data))

model = tf.keras.Sequential()
model.add(layers.Dense(150,activation=activations.relu, input_shape=(15,)))
model.add(layers.Dense(100,activation=activations.relu))
model.add(layers.Dense(100,activation=activations.relu))
model.add(layers.Dense(1))
model.weights
model.summary()

batch_size = 128
epochs = 10

model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(traj_data[:,:15], traj_data[:,15], batch_size=batch_size, epochs=epochs)
feed = np.array([traj_data[90,:15]])
print(model.predict(feed))

# print(np.average(1-np.absolute(traj_data[:,15] - model.predict(traj_data[:,:15])/traj_data[:,15])))

# model.save('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/weights/initialize')

# model2 = tf.keras.models.load_model('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/weights/initialize')
# feed = np.array([traj_data[0]])
# feed = feed.reshape(-1,1)
# print(np.shape(feed))
# print(np.average(1-np.absolute(val_data - model2.predict(traj_data)/val_data)))
# print(model2.predict(feed))
'''
with open('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/trajectory_dataset/2_agents/trajs/RVO5.pkl', 'rb') as f:
    data = pickle.load(f)
gamma = 0.98

traj_arr = []
val_arr = []
for traj in data:
    if traj != []:
        traj = np.array(traj)
        traj_arr.append(traj[:,:15]) #Comment out when using NORMALIZED
        # traj_arr.append(traj[:,:15]/np.array([17.32,2,2,2,0.8,3.15,2,2,15.2,13.8,0.8,1.6,1,1,16])) # NORMALIZED rvo2
        # traj_arr.append(traj[:,:15]/np.array([17.32,2,2,2,0.8,6.3,2,2,15.2,13.8,0.8,1.6,1,1,16])) # NORMALIZED rvo3 0to360
        val_arr.append(gamma**(traj[:,15]*traj[:,1]))
traj_data = traj_arr[0]
val_data = val_arr[0]
for i in range(len(traj_arr)):
    traj_data = np.vstack((traj_data,traj_arr[i]))
    val_data = np.hstack((val_data,val_arr[i]))
val_data = val_data.reshape(-1,1)
'''