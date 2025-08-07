import sys
import time
import math
import pickle

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from library.data.pipeline import batch_generator
from library.models.nn import get_3d_groundwater_flow_model, sample_3d_model
from library.models.util import fit
from library.models.metrics import count_params
from library.visual import plot_surface3d, animate_hydrology


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
I_CACHE = 'cache/data_interpolated.npz'
S_CACHE = 'cache/data_surface.csv'
H_CACHE = 'cache/gs_as_cache.pkl'
CACHE_ENABLED = True

# RNG setup
RNG_SEED = 999
K0, K1, K2 = jax.random.split(jax.random.key(RNG_SEED), 3)

# data
TRAIN_EPOCH = 2
TRAIN_BATCH = 64
PART_TRAIN = 0.75
PART_VAL = 0.05
PART_TEST = 0.20

# model
MDL_LAYERS = [3, 256, 256, 1] ###![45]
MDL_ACTIVATION = lambda x,s: jax.nn.tanh(x*s)
MDL_ACTIVATION_SCALE_MIN = 1.0#1e-3
MDL_ACTIVATION_SCALE_MAX = 10#1.0
MDL_ACTIVATION_SCALE_RES = 8

###![45]
###! enlarging param space yields no effect
#MDL_ACTIVATION = jax.nn.relu # learns localized curvature, not twice differentiable
#MDL_ACTIVATION = jax.nn.leaky_relu # same as relu
###! twice differentiable but physics loss vanished anyway
#MDL_ACTIVATION = lambda x: jax.nn.tanh(x*6) # learns global pattern, no localised curvature
#MDL_ACTIVATION = jnp.sin # as above
#MDL_ACTIVATION = lambda x: jnp.log1p(jnp.exp(x)) # as above
#MDL_ACTIVATION = jax.nn.sigmoid # flat through mean

# loss
LAM_MSE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
LAM_PHYS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LAM_L2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
LAM_SS = 1e-5
LAM_RR = 0.0

# optimizer
OPT = optax.adamw
OPT_ETA = 1e-4

# sampling
SAMPLE_ACTIVATION = jnp.linspace(-5, 5, 32)
SAMPLE_3D_XMIN = 0#-2
SAMPLE_3D_XMAX = 1#+3
SAMPLE_3D_XRES = 0 ###! 0 -> inherit k shape
SAMPLE_3D_YMIN = 0#-2
SAMPLE_3D_YMAX = 1#+3
SAMPLE_3D_YRES = 0 ###! 0 -> inherit k shape
SAMPLE_3D_TMIN = 0
SAMPLE_3D_TMAX = 2.6
SAMPLE_3D_TRES = 260
SAMPLE_3D_BATCH = False


### main

# load cache
with jnp.load(I_CACHE) as i_cache:
	k_crop = i_cache['k_crop']
	data_wells = i_cache['data_wells']
	print(f"Loaded \"{I_CACHE}\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

data_surface = pd.read_csv(S_CACHE).to_numpy()
print(f"Loaded \"{S_CACHE}\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# populate xyt->z
data_points = []
for i in range(0, data_surface.shape[0], 1):
	for j in range(0, data_surface.shape[1]-1, 1):
		xytz = (*data_wells[j], data_surface[i][0], data_surface[i][j+1]) # xytz
		data_points.append(xytz)

data_points = jnp.array(data_points)

###! raw measurements
# M_CACHE = 'cache/data_filtered_metric.csv'
# data_filtered_metric = pd.read_csv(M_CACHE)
# data_filtered_metric = data_filtered_metric.dropna()
# data_points = data_filtered_metric[['X_EPSG_6350', 'Y_EPSG_6350', 'TIMESTAMP', 'HYDRAULIC_HEAD_M']].to_numpy()

# partition data
# n_data = len(data_points)
# shuffle_idx = jax.random.permutation(K0, n_data)
# data_train = data_points[shuffle_idx[:int(PART_TRAIN * n_data)]]
# data_val = data_points[shuffle_idx[int(PART_TRAIN * n_data) : int((PART_TRAIN + PART_VAL) * n_data)]]
# data_test = data_points[shuffle_idx[int((PART_TRAIN + PART_VAL) * n_data) : int((PART_TRAIN + PART_VAL + PART_TEST) * n_data)]]

###! partition data with train/val-test split in time order, shuffling val and test together
n_data = data_points.shape[0]
n_train = math.floor(PART_TRAIN * n_data)
n_val = math.floor(PART_VAL * n_data)
n_test = math.floor(PART_TEST * n_data)
shuffle_idx = n_train + jax.random.permutation(K0, n_val + n_test)
data_train = data_points[:n_train]
data_val = data_points[shuffle_idx[:n_val]]
data_test = data_points[shuffle_idx[n_val:n_test]]

# project to unit hypercube
data_scaler = MinMaxScaler(feature_range=(0, 1))
data_scaler.fit(data_train)
data_scale_xytz = data_scaler.data_range_

data_train = data_scaler.transform(data_train)
data_val = data_scaler.transform(data_val)
data_test = data_scaler.transform(data_test)

# supervised split
train_x, train_y = data_train[:,:-1], data_train[:,-1] # xyt, z
val_x, val_y = data_val[:,:-1], data_val[:,-1]
test_x, test_y = data_test[:,:-1], data_test[:,-1]

# determine batch counts
train_steps = math.ceil(train_x.shape[0] / TRAIN_BATCH)
val_steps = math.ceil(val_x.shape[0] / TRAIN_BATCH)
test_steps = math.ceil(test_x.shape[0] / TRAIN_BATCH)

# memory cleanup
del shuffle_idx
del data_points
del data_train
del data_val
del data_test

# 0 1 1
# 0 0 1
# 1 1 0
# 0 0 0

# train_x = jnp.array([[0,1,0],[0,0,0],[1,1,0],[0,0,0]])
# train_y = jnp.array([[1],[1],[0],[0]])
# train_steps = 1

# val_x = jnp.array([[0,1,0],[0,0,0],[1,1,0],[0,0,0]])
# val_y = jnp.array([[1],[1],[0],[0]])
# val_steps = 1

# test_x = jnp.array([[0,1,0],[0,0,0],[1,1,0],[0,0,0]])
# test_y = jnp.array([[1],[1],[0],[0]])
# test_steps = 1

# data_scale_xytz = jnp.ones(4)

# trace
print(f"Train: x~{train_x.shape}, y~{train_y.shape}, steps={train_steps}")
print(f"Val: x~{val_x.shape}, y~{val_y.shape}, steps={val_steps}")
print(f"Test: x~{test_x.shape}, y~{test_y.shape}, steps={test_steps}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")


###! gridsearch

try:
	with open(H_CACHE, 'rb') as f:
		h_cache = pickle.load(f)
		data_scaler=h_cache['data_scaler']
		params=h_cache['params']
		trial_axis=h_cache['trial_axis']
		trial_history=h_cache['trial_history']
		print(f"Loaded \"{H_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception:
	trial_axis = jnp.linspace(MDL_ACTIVATION_SCALE_MIN, MDL_ACTIVATION_SCALE_MAX, MDL_ACTIVATION_SCALE_RES)
	trial_history = []
	for activation_scale in trial_axis:
		
		# init model
		trial_activation = lambda x: MDL_ACTIVATION(x, activation_scale)
		params, h_fn, loss_fn = get_3d_groundwater_flow_model(
			K1,
			MDL_LAYERS,
			scale_xytz=data_scale_xytz,
			k=k_crop,
			ss=LAM_SS,
			rr=LAM_RR,
			lam_mse=LAM_MSE,
			lam_phys=LAM_PHYS,
			lam_l2=LAM_L2,
			hidden_activation=trial_activation
		)
		trial_activation_sample = trial_activation(SAMPLE_ACTIVATION)
		print(f"activation_scale={activation_scale}, count_params(params)={count_params(params)}, trial_activation_sample={trial_activation_sample}")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		# fit model
		params, history = fit(
			K2,
			params,
			loss_fn,
			(train_x, train_y, train_steps),
			val_data=(val_x, val_y, val_steps),
			batch_size=TRAIN_BATCH,
			epochs=TRAIN_EPOCH,
			opt=OPT(OPT_ETA),
			start_time=T0
		)
		print(f"LAM_SS={float(params[-1][0])}, LAM_RR={float(params[-1][1])}")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		# test model
		test_generator = batch_generator(test_x, test_y, TRAIN_BATCH)
		test_loss = 0.
		for _ in range(test_steps):
			test_loss += loss_fn(params, *next(test_generator)) / test_steps
		print(f"test_loss={test_loss:.4f}")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		# sample model
		axis_x = jnp.linspace(SAMPLE_3D_XMIN, SAMPLE_3D_XMAX, k_crop.shape[1] if (SAMPLE_3D_XRES==0) else SAMPLE_3D_XRES)
		axis_y = jnp.linspace(SAMPLE_3D_YMIN, SAMPLE_3D_YMAX, k_crop.shape[0] if (SAMPLE_3D_YRES==0) else SAMPLE_3D_YRES)
		axis_t = jnp.linspace(SAMPLE_3D_TMIN, SAMPLE_3D_TMAX, SAMPLE_3D_TRES)
		h_sim = sample_3d_model(h_fn, params[0], axis_t, axis_y, axis_x, batch_size=None)
		h_sim = data_scaler.data_min_[3] + h_sim * data_scaler.data_range_[3]
		print(f"h_sim.shape={h_sim.shape}")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		# log
		history['test_loss'] = [test_loss]
		history['sampling'] = {}
		history['sampling']['activation_range'] = SAMPLE_ACTIVATION
		history['sampling']['activation'] = trial_activation_sample
		history['sampling'] = {'surface':dict(
			axis_x=axis_x,
			axis_y=axis_y,
			axis_t=axis_t,
			h_sim=h_sim
		)}
		trial_history.append(history)
		
		# create cache
		if CACHE_ENABLED:
			with open(H_CACHE, 'wb') as f:
				h_cache = dict(
					data_scaler=data_scaler,
					params=params,
					trial_axis=trial_axis,
					trial_history=trial_history
				)
				pickle.dump(h_cache, f)
				print(f"Saved \"{H_CACHE}\"")
				print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(trial_axis, [h['test_loss'] for h in trial_history], c='green')
ax.set_xlabel("activation_scale")
ax.set_ylabel("Test loss")
plt.show()
