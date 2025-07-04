import time
import math
import pickle

import jax
import jax.numpy as jnp
import optax

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from library.models.nn import *
from library.visual import *


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
I_CACHE = 'cache/data_interpolated.npz'
S_CACHE = 'cache/data_surface.csv'
G_CACHE = 'cache/cache_gs_pt.pkl'

# RNG setup
RNG_SEED = 999
K0, K1, K2 = jax.random.split(jax.random.key(RNG_SEED), 3)

# data partitions
EPOCHS = 2
BATCH_SIZE = 64
PART_VAL = 0.05

# optimizer
ETA = 1e-4
LAM_MSE = 1.0
LAM_PHYS = 1.0
LAM_L2 = 0.0

# physical constants (default)
SS = 1e-1
RR = 0.0


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

# try cache
try:
	with open(G_CACHE, 'rb') as f:
		axis_part, axis_history = pickle.load(f)
		print(f"Loaded \"{G_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception as e:
	# gridsearch over train ratio in 5% increments, fixing val ratio to 5%
	# and alloting remaining portion to test set, ensuring no set is ever 
	# less than 5% of the full set
	axis_part = jnp.arange(0.05, 0.95, 0.05)
	axis_history = []
	for part_train in axis_part:
		part_test = 1 - PART_VAL - part_train
		
		# partition data with train/val-test split in time order, shuffling val and test together
		n_data = data_points.shape[0]
		n_train = int(part_train * n_data)
		n_val = int(PART_VAL * n_data)
		n_test = int(part_test * n_data)
		shuffle_idx = n_train + jax.random.permutation(K0, n_val + n_test)
		data_train = data_points[:n_train]
		data_val = data_points[shuffle_idx[:n_val]]
		data_test = data_points[shuffle_idx[n_val:]]
		
		# project to unit hypercube
		data_scaler = MinMaxScaler(feature_range=(0, 1))
		data_scaler.fit(data_train)
		data_scale_xytz = data_scaler.data_range_ / jnp.array([1., 1., 3600, 1.]) # units (m, m, hr, m)
		
		data_train = data_scaler.transform(data_train)
		data_val = data_scaler.transform(data_val)
		data_test = data_scaler.transform(data_test)
		
		# supervised split
		train_x, train_y = data_train[:,:-1], data_train[:,-1] # xyt, z
		val_x, val_y = data_val[:,:-1], data_val[:,-1]
		test_x, test_y = data_test[:,:-1], data_test[:,-1]
		
		# determine batch counts
		train_steps = math.ceil(train_x.shape[0] / BATCH_SIZE)
		val_steps = math.ceil(val_x.shape[0] / BATCH_SIZE)
		test_steps = math.ceil(test_x.shape[0] / BATCH_SIZE)
		
		# memory cleanup
		del shuffle_idx
		del data_train
		del data_val
		del data_test
		
		# trace
		print(f"Train: x~{train_x.shape}, y~{train_y.shape}, steps={train_steps}")
		print(f"Val: x~{val_x.shape}, y~{val_y.shape}, steps={val_steps}")
		print(f"Test: x~{test_x.shape}, y~{test_y.shape}, steps={test_steps}")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		# initialise model+loss
		params, h_fn, loss_fn = get_3d_groundwater_flow_model(
			K1,
			[3, 256, 256, 1],
			data_scale_xytz,
			k=k_crop,
			ss=SS,
			rr=RR,
			lam_mse=LAM_MSE,
			lam_phys=LAM_PHYS,
			lam_l2=LAM_L2
		)
		
		# fit model
		params, history = fit_model(
			K2,
			params,
			loss_fn,
			(train_x, train_y, train_steps),
			val_data=(val_x, val_y, val_steps),
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			opt=optax.adamw(ETA),
			start_time=T0
		)
		print(f"SS={float(params[-1][0])}, RR={float(params[-1][1])}")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		# evaluate model
		###!
		#test_generator = batch_generator(test_x, test_y, BATCH_SIZE)
		#test_loss = 0.
		#for _ in range(test_steps):
		#	test_loss += loss_fn(params, *next(test_generator)) / test_steps
		###!
		test_loss = loss_fn(params, test_x, test_y)
		test_rmse = loss_mse(h_fn(params[0], test_x), test_y) ** 0.5
		
		history['test_loss'] = [test_loss]
		history['test_rmse'] = [test_rmse]
		print(f"test_loss={test_loss:.4f}")
		print(f"test_rmse={test_rmse:.4f}")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		axis_history.append(history)
		print(f"*** Completed trial: part_train={part_train:.2f} ***")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		# create cache
		with open(G_CACHE, 'wb') as f:
			pickle.dump((axis_part, axis_history), f)
			print(f"Saved \"{G_CACHE}\"")
			print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot test loss
n_days = len(data_surface)
val_days = int(PART_VAL * n_days)

axis_part_days = jnp.astype(n_days * axis_part, 'int32')
axis_test_rmse = [history['test_rmse'] for history in axis_history]

plt.plot(axis_part_days, axis_test_rmse, c='darkgreen')
plt.xlabel("#Days (Train:Test)")
plt.ylabel("Test RMSE (Metres)")
plt.xticks(axis_part_days[::3], [f"{train_days}:{n_days-val_days-train_days}" for train_days in axis_part_days[::3]])
plt.grid()
plt.show()
