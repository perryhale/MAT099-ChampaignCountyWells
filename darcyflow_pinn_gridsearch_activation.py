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
MODEL_LAYERS = [3, 256, 256, 1]
MODEL_ACTIVATION = lambda a,b,x: jax.nn.tanh(a*x) + b*a*x*jax.nn.tanh(x) # stan squash
MODEL_ACTIVATION_A_MIN = 1
MODEL_ACTIVATION_A_MAX = 5
MODEL_ACTIVATION_A_RES = 8
MODEL_ACTIVATION_B_MIN = 0
MODEL_ACTIVATION_B_MAX = 5
MODEL_ACTIVATION_B_RES = 8

# loss terms
LAM_MSE = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
LAM_PHYS = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LAM_L2 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
LAM_SS = 1e-5
LAM_RR = 0.0

# optimizer params
OPT = optax.adamw
OPT_ETA = 1e-4

# all sampling
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
SAMPLE_ACTIVATION = jnp.linspace(-2, 2, 32)


### functions

def trial_fn(a,b,
		data_scale_xytz, k_crop, train_x, train_y, train_steps, val_x, val_y, val_steps, test_x, test_y, test_steps
	):
	
	# init model
	trial_activation = lambda x: MODEL_ACTIVATION(a, b, x)
	params, h_fn, loss_fn = get_3d_groundwater_flow_model(
		K1,
		MODEL_LAYERS,
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
	print(f"a={a}, b={b}, count_params(params)={count_params(params)}, trial_activation_sample={trial_activation_sample}")
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
	
	# record stats
	history['test_loss'] = [test_loss]
	history['sample'] = {}
	history['sample']['activation_range'] = SAMPLE_ACTIVATION
	history['sample']['activation'] = trial_activation_sample
	history['sample']['surface'] = dict(
		axis_x=axis_x,
		axis_y=axis_y,
		axis_t=axis_t,
		h_sim=h_sim
	)
	history['params'] = params
	
	return history


### main

# load caches
with jnp.load(I_CACHE) as i_cache:
	k_crop = i_cache['k_crop']
	data_wells = i_cache['data_wells']
print(f"Loaded \"{I_CACHE}\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

data_surface = pd.read_csv(S_CACHE).to_numpy()
print(f"Loaded \"{S_CACHE}\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# populate xytz samples
data_points = jnp.array([[
	data_wells[j][0],
	data_wells[j][1],
	data_surface[i][0],
	data_surface[i][j+1]
] for j in range(0, data_surface.shape[1]-1, 1) for i in range(0, data_surface.shape[0], 1)])[:2000]

# partition data
n_data = data_points.shape[0]
n_train = math.floor(PART_TRAIN * n_data)
n_val = math.floor(PART_VAL * n_data)
n_test = math.floor(PART_TEST * n_data)
shuffle_idx = n_train + jax.random.permutation(K0, n_val + n_test)
data_train = data_points[:n_train]
data_val = data_points[shuffle_idx[:n_val]]
data_test = data_points[shuffle_idx[n_val:n_test]]

# normalise data
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

# trace
print(f"Train: x~{train_x.shape}, y~{train_y.shape}, steps={train_steps}")
print(f"Val: x~{val_x.shape}, y~{val_y.shape}, steps={val_steps}")
print(f"Test: x~{test_x.shape}, y~{test_y.shape}, steps={test_steps}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# 2d gridsearch
try:
	with open(H_CACHE, 'rb') as f:
		h_cache = pickle.load(f)
		data_scaler=h_cache['data_scaler']
		a_axis=h_cache['a_axis']
		b_axis=h_cache['b_axis']
		results=h_cache['results']
		print(f"Loaded \"{H_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception:
	
	a_axis = jnp.linspace(MODEL_ACTIVATION_A_MIN, MODEL_ACTIVATION_A_MAX, MODEL_ACTIVATION_A_RES)
	b_axis = jnp.linspace(MODEL_ACTIVATION_B_MIN, MODEL_ACTIVATION_B_MAX, MODEL_ACTIVATION_B_RES)
	results = [[None]*len(b_axis)]*len(a_axis)#jnp.empty((a_axis.shape[0], b_axis.shape[0]), dtype='object')
	
	for i,a in enumerate(a_axis):
		for j,b in enumerate(b_axis):
			
			# run trial
			history = trial_fn(a,b,
				data_scale_xytz, k_crop, train_x, train_y, train_steps, val_x, val_y, val_steps, test_x, test_y, test_steps
			)
			results[i][j] = history
			
			# cache checkpoint
			if CACHE_ENABLED:
				with open(H_CACHE, 'wb') as f:
					h_cache = dict(
						data_scaler=data_scaler,
						a_axis=a_axis,
						b_axis=b_axis,
						results=results
					)
					pickle.dump(h_cache, f)
					print(f"Saved \"{H_CACHE}\"")
					print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot
