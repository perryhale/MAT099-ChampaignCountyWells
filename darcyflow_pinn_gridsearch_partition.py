import time
import math
import pickle

import jax
import jax.numpy as jnp
import optax

import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('classic')
import matplotlib.patches as patches
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
PART_TEST = 0.25

# optimizer
ETA = 1e-4
LAM_MSE = 1.0
LAM_PHYS = 1.0
LAM_L2 = 0.0

# physical constants (default)
SS = 1e-5
RR = 0.0


### functions

def trial_fn(data_points, part_train, k0, batch_size, k1, k_crop, ss, rr, lam_mse, lam_phys, lam_l2, k2, epochs, eta):
	
	part_buffer = 1 - part_train - PART_VAL - PART_TEST
	assert (part_buffer >= 0 or part_buffer <= 1)
	
	# partition data with train/val-test split in time order, shuffling
	# val and test together - cutting train from earliest samples and
	# val+test from the latest samples.
	n_data = data_points.shape[0]
	n_train = int(part_train * n_data)
	n_val = int(PART_VAL * n_data)
	n_test = int(PART_TEST * n_data)
	n_buffer = int(part_buffer * n_data)
	
	shuffle_idx = n_train + n_buffer + jax.random.permutation(k0, n_val + n_test)
	data_train = data_points[:n_train]
	data_val = data_points[shuffle_idx[:n_val]]
	data_test = data_points[shuffle_idx[n_val:]]
	
	# unit scale
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
	train_steps = math.ceil(train_x.shape[0] / batch_size)
	val_steps = math.ceil(val_x.shape[0] / batch_size)
	test_steps = math.ceil(test_x.shape[0] / batch_size)
	
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
		k1, [3, 256, 256, 1],
		data_scale_xytz, k_crop, ss, rr,
		lam_mse, lam_phys, lam_l2
	)
	print(f"count_params(params)={count_params(params)}")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# fit model
	params, history = fit_model(
		k2, params, loss_fn,
		(train_x, train_y, train_steps), (val_x, val_y, val_steps),
		batch_size, epochs, optax.adamw(eta), T0
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
	
	return history


def plot_data_partitions(part_train, title="", scale=1):
	
	part_buffer = 1 - part_train - PART_VAL - PART_TEST
	assert 0 <= part_buffer <= 1
	
	lefts = [l*scale for l in [0,
			 part_train,
			 part_train + part_buffer,
			 part_train + part_buffer + PART_VAL]]
	widths = [l*scale for l in [part_train, part_buffer, PART_VAL, PART_TEST]]
	colors = ['blue', 'grey', 'red', 'green']
	labels = ['Train', 'Buffer', 'Val', 'Test']
	
	fig, ax = plt.subplots(figsize=(8, 3))
	
	for left, width, color, label in list(zip(lefts, widths, colors, labels))[:2]:
		if width > 0:
			ax.add_patch(patches.Rectangle((left, 0), width, scale, color=color, label=label))
	
	ax.add_patch(patches.Rectangle((lefts[2], 0), sum(widths[2:]), scale, label='+'.join(labels[2:]),
		facecolor=colors[3],
		edgecolor=colors[2],
		hatch='/',
		linewidth=1
	))
	
	ax.set_xlim(0, scale)
	ax.set_ylim(0, scale)
	ax.set_yticks([])
	ax.set_xticks([l*scale for l in [0, part_train, part_train + part_buffer, 1]])
	ax.set_xlabel(title)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
	
	plt.tight_layout()
	plt.show()


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

# try cache
try:
	with open(G_CACHE, 'rb') as f:
		axis_part, axis_history = pickle.load(f)
		print(f"Loaded \"{G_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception as e:
	
	# populate data_points
	data_points = []
	for i in range(0, data_surface.shape[0], 1):
		for j in range(0, data_surface.shape[1]-1, 1):
			xytz = (*data_wells[j], data_surface[i][0], data_surface[i][j+1]) # xytz
			data_points.append(xytz)
	
	data_points = jnp.array(data_points)
	
	# define trial axis
	axis_part = jnp.arange(0.05, 0.75, 0.05)
	axis_history = []
	
	# determine losses for each trial with physics loss
	for part_train in axis_part:
		history = trial_fn(
			data_points, part_train, K0, BATCH_SIZE, K1, k_crop, SS, RR, LAM_MSE, LAM_PHYS, LAM_L2, K2, EPOCHS, ETA
		)
		axis_history.append(history)
		print(f"*** Completed trial: part_train={part_train:.2f} ***")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		with open(G_CACHE, 'wb') as f:
			pickle.dump((axis_part, axis_history), f)
			print(f"Saved \"{G_CACHE}\"")
			print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# determine losses for each trial without physics loss
	for part_train in axis_part:
		history = trial_fn(
			data_points, part_train, K0, BATCH_SIZE, K1, k_crop, SS, RR, LAM_MSE, 0.0, LAM_L2, K2, EPOCHS, ETA
		)
		axis_history.append(history)
		print(f"*** Completed trial: part_train={part_train:.2f} ***")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		
		with open(G_CACHE, 'wb') as f:
			pickle.dump((axis_part, axis_history), f)
			print(f"Saved \"{G_CACHE}\"")
			print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot rmse over part_train
n_trials = len(axis_part)
n_days = len(data_surface)
axis_part_days = jnp.astype(n_days * axis_part, 'int32')
val_days = int(PART_VAL * n_days)
test_days = int(PART_TEST * n_days)
axis_test_rmse = [history['test_rmse'] for history in axis_history]

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(axis_part_days, axis_test_rmse[:n_trials], c='darkgreen', label="PINN")
ax.plot(axis_part_days, axis_test_rmse[n_trials:], c='darkred', linestyle='dashed', label="MSE")
ax.legend()
ax.set_ylabel("Test RMSE (Metres)")
ax.set_xlabel("Num. days in [Train:Test] sets")
ax.set_xticks(axis_part_days[::3], [f"{train_days}:{test_days}" for train_days in axis_part_days[::3]])
ax.grid()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot partition scheme
trial_idx = 4
plot_data_partitions(axis_part[trial_idx], title=f"Data partition in days\nTest RMSE={axis_test_rmse[trial_idx][0]:.2f}m", scale=n_days)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
