import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras import activations
from keras.models import Sequential
from keras.models import Dense

with open('/home/atharva/cadrl2/gym-collision-avoidance/gym_collision_avoidance/experiments/results/trajectory_dataset/2_agents/trajs/RVO2.pkl', 'rb') as f:
    data = pickle.load(f)
gamma = 0.98

traj_arr = []
val_arr = []
for traj in data:
    if traj != []:
        traj = np.array(traj)
        traj_arr.append(traj[:,:15]) #Comment out when using NORMALIZED
        # traj_arr.append(traj[:,:15]/np.array([20,2,2,2,0.8,3.15,2,2,20,20,0.8,1.6,1,1,20])) # NORMALIZED
        val_arr.append(gamma**(traj[:,15]*traj[:,1]))
traj_data = traj_arr[0]
val_data = val_arr[0]
for i in range(len(traj_arr)):
    traj_data = np.vstack((traj_data,traj_arr[i]))
    val_data = np.hstack((val_data,val_arr[i]))
val_data = val_data.reshape(-1,1)


model = Sequential()
model.add(Dense(150,activation='relu', input_dim=(15)))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(1, activation='linear'))
# model.weights
model.summary()

batch_size = 100
epochs = 15

# train_dataset = tf.data.Dataset.from_tensor_slices((traj_data, val_data))

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy"])

model.fit(traj_data, val_data, batch_size=batch_size, epochs=epochs)
# model.fit(train_dataset, epochs=epochs)