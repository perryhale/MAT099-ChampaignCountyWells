import time
import math
import pickle

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('classic')
from matplotlib.patches import Rectangle
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from library.models.nn import get_3d_groundwater_flow_model
from library.models.metrics import count_params
from library.models.util import fit


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

# trial axis
AX_LAM_PHYS = jnp.array([LAM_PHYS, 0.0])
AX_PART_TRAIN = jnp.arange(0.05, 0.75, 0.05)
AX_PART_BUFFER_FN = lambda part_train: jnp.arange(0.00, max(0, 1 - part_train - PART_VAL - PART_TEST)+0.05, 0.05)


### functions

def trial_fn(data_points, part_buffer, part_train, k0, batch_size, k1, k_crop, ss, rr, lam_mse, lam_phys, lam_l2, k2, epochs, eta):
	
	# partition respecting time-series, shuffle val+test together
	n_data = data_points.shape[0]
	n_buffer = int(part_buffer * n_data)
	n_train = int(part_train * n_data)
	n_val = int(PART_VAL * n_data)
	n_test = int(PART_TEST * n_data)
	
	shuffle_idx = n_buffer + n_train + jax.random.permutation(k0, n_val + n_test)
	data_train = data_points[n_buffer:n_buffer+n_train]
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
	params, history = fit(
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


def plot_data_partition(part_buffer, part_train, title="", scale=1, shuffle_val_test=True, buffer_all=True):
	
	lefts = [l*scale for l in [
		0,
		part_buffer,
		part_buffer + part_train,
		part_buffer + part_train + PART_VAL
	]]
	widths = [l*scale for l in [
		part_buffer,
		part_train,
		PART_VAL,
		PART_TEST
	]]
	colors = ['grey', 'blue', 'red', 'green']
	labels = ['Buffer', 'Train', 'Val', 'Test']
	
	fig, ax = plt.subplots(figsize=(8, 3))
	ax.add_patch(Rectangle((lefts[0], 0), scale if buffer_all else widths[0], scale, color=colors[0], label=labels[0]))
	ax.add_patch(Rectangle((lefts[1], 0), widths[1], scale, color=colors[1], label=labels[1]))
	if shuffle_val_test:
		ax.add_patch(Rectangle((lefts[2], 0), sum(widths[2:]), scale, label='+'.join(labels[2:]),
			facecolor=colors[3],
			edgecolor=colors[2],
			hatch='/',
			linewidth=1
		))
	else:
		ax.add_patch(Rectangle((lefts[2], 0), widths[2], scale, color=colors[2], label=labels[2]))
		ax.add_patch(Rectangle((lefts[3], 0), widths[3], scale, color=colors[3], label=labels[3]))
	
	ax.set_xlim(0, scale)
	ax.set_ylim(0, scale)
	ax.set_yticks([])
	ax.set_xticks([l*scale for l in [0, part_buffer, part_buffer + part_train, 1]])#, [f"+{w:n}" for w in [0, widths[0], widths[1], sum(widths[2:])]])
	ax.set_xlabel(title)
	ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4)
	
	return fig, ax


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
		trial_history_ax0 = pickle.load(f)
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
	
	# determine results and store as linked list
	trial_history_ax0 = []
	for lam_phys in AX_LAM_PHYS:
		
		trial_history_ax1 = []
		for part_train in AX_PART_TRAIN:
			
			trial_history_ax2 = []
			for part_buffer in AX_PART_BUFFER_FN(part_train):
				
				print(f"*** Trial: lam_phys={lam_phys:.2f}, part_train={part_train:.2f}, part_buffer={part_buffer:.2f} ***")
				history = trial_fn(data_points, part_buffer, part_train, K0, BATCH_SIZE, K1, k_crop, SS, RR, LAM_MSE, lam_phys, LAM_L2, K2, EPOCHS, ETA)
				history['metadata'] = dict(
					lam_phys=lam_phys,
					part_train=part_train,
					part_buffer=part_buffer
				)
				print(f"[Elapsed time: {time.time()-T0:.2f}s]")
				
				trial_history_ax2.append(history)
			
			trial_history_ax1.append(trial_history_ax2)
		
		trial_history_ax0.append(trial_history_ax1)
		
		# save cache
		with open(G_CACHE, 'wb') as f:
			pickle.dump(trial_history_ax0, f)
			print(f"Saved \"{G_CACHE}\"")
			print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# cleanup
	del data_points
	del trial_history_ax1
	del trial_history_ax2
	del history


# plot rmse over part_train
n_days = len(data_surface)
test_days = int(PART_TEST * n_days)
axis_train_days = jnp.astype(n_days * AX_PART_TRAIN, 'int32')

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(axis_train_days, [sum([h['test_rmse'][0] for h in h_ax2]) / len(h_ax2) for h_ax2 in trial_history_ax0[0]], c='darkgreen', label="PINN")
ax.plot(axis_train_days, [sum([h['test_rmse'][0] for h in h_ax2]) / len(h_ax2) for h_ax2 in trial_history_ax0[1]], c='darkred', linestyle='dashed', label="MSE")
ax.legend()
ax.set_ylabel("Cross-validated test RMSE (Metres)")
ax.set_xlabel("Num. days in [Train:Test] sets")
ax.set_xticks(axis_train_days[::3], [f"{train_days}:{test_days}" for train_days in axis_train_days[::3]])
ax.grid()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot all partition schemes
frame_counter = 0
for i, part_train in enumerate(AX_PART_TRAIN):
	for j, part_buffer in enumerate(AX_PART_BUFFER_FN(part_train)):
		
		test_rmse = trial_history_ax0[0][i][j]['test_rmse'][0]
		output_name = f"{__file__.replace('.py','')}_Figure_2_{frame_counter}.png"
		
		plot_data_partition(part_buffer, part_train, title=f"Data partition in days\nTest RMSE={test_rmse:.2f}m", scale=n_days)
		plt.tight_layout()
		plt.savefig(output_name)
		plt.close()
		frame_counter += 1
		print(f"Saved \"{output_name}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
