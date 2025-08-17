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

from library.data.pipeline import train_val_test_split
from library.models.nn import get_3d_groundwater_flow_model
from library.models.util import fit, count_params, loss_mse


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# RNG setup
RNG_SEED = 999
K0, K1, K2 = jax.random.split(jax.random.key(RNG_SEED), 3)

# cache path
I_CACHE = 'cache/data_interpolated.npz'
S_CACHE = 'cache/data_surface.csv'
G_CACHE = 'cache/cache_gs_pt.pkl'

# model
MODEL_LAYERS = [3, 256, 256, 1]

###! variations
MODEL_ACTIVATION = lambda x: jax.nn.tanh(x) + 3.9 * x * jax.nn.tanh(x) # optimal stan
#MODEL_ACTIVATION = lambda x: jax.nn.tanh(3.57 * x) # optimal squashed tanh
#MODEL_ACTIVATION = lambda x: jax.nn.relu(6.1 * x) # optimal squashed relu
#MODEL_ACTIVATION = jax.nn.relu # normal relu

# loss terms
LAM_MSE = 1.0
LAM_PHYS = 1.0
LAM_L2 = 0.0
SS = 1e-5
RR = 0.0

# optimizer
ETA = 1e-4
EPOCHS = 2
BATCH_SIZE = 64

# data partition
PART_VAL = 0.05
PART_TEST = 0.25
AX_LAM_PHYS = jnp.array([LAM_PHYS, 0.0])
AX_PART_TRAIN = jnp.arange(0.05, 0.75, 0.05)
AX_PART_BUFFER_FN = lambda part_train: jnp.arange(0.00, max(0, 1 - part_train - PART_VAL - PART_TEST)+0.05, 0.05)

# plotting args
PLOT_SCHEMES = False


### functions

def trial_fn(part_buffer, part_train, lam_phys, data_points, k_crop):
	
	# prepare data
	data_split = train_val_test_split(K0, data_points, BATCH_SIZE, part_buffer=part_buffer, part_train=part_train, part_val=PART_VAL, part_test=PART_TEST)
	(train_x, train_y, train_steps), (val_x, val_y, val_steps), (test_x, test_y, test_steps), data_scaler = data_split
	data_scale_xytz = data_scaler.data_range_ / jnp.ones(len(data_scaler.data_range_), dtype='float32') # units (m, m, s, m)
	print(f"Train: x~{train_x.shape}, y~{train_y.shape}, steps={train_steps}")
	print(f"Val: x~{val_x.shape}, y~{val_y.shape}, steps={val_steps}")
	print(f"Test: x~{test_x.shape}, y~{test_y.shape}, steps={test_steps}")
	print(f"Scale: [{', '.join([f'{float(scale.item()):.1f}{unit}' for scale, unit in zip(data_scale_xytz, 'mmsm')])}]")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# initialise model+loss
	params, h_fn, loss_fn = get_3d_groundwater_flow_model(
		K1,
		MODEL_LAYERS,
		scale_xytz=data_scale_xytz,
		k=k_crop,
		ss=SS,
		rr=RR,
		lam_mse=LAM_MSE,
		lam_phys=lam_phys,
		lam_l2=LAM_L2,
		hidden_activation=MODEL_ACTIVATION
	)
	print(f"count_params(params)={count_params(params)}")
	
	# fit model
	params, history = fit(
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
	print(f"ss={float(params[-1][0])}, rr={float(params[-1][1])}")
	
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

# load caches
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
		results = pickle.load(f)
		print(f"Loaded \"{G_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception as e:
	
	# populate data_points
	data_points = jnp.array([(*data_wells[j], data_surface[i][0], data_surface[i][j+1]) for i in range(data_surface.shape[0]) for j in range(data_surface.shape[1]-1)])
	
	# determine results and store as linked list
	results = []
	for lam_phys in AX_LAM_PHYS:
		
		results_ax1 = []
		for part_train in AX_PART_TRAIN:
			
			results_ax2 = []
			for part_buffer in AX_PART_BUFFER_FN(part_train):
				
				print(f"*** Trial: lam_phys={lam_phys:.2f}, part_train={part_train:.2f}, part_buffer={part_buffer:.2f} ***")
				history = trial_fn(part_buffer, part_train, lam_phys, data_points, k_crop)
				history['metadata'] = dict(
					lam_phys=lam_phys,
					part_train=part_train,
					part_buffer=part_buffer
				)
				print(f"[Elapsed time: {time.time()-T0:.2f}s]")
				
				results_ax2.append(history)
			
			results_ax1.append(results_ax2)
		
		results.append(results_ax1)
		
		# save cache
		with open(G_CACHE, 'wb') as f:
			pickle.dump(results, f)
			print(f"Saved \"{G_CACHE}\"")
			print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# cleanup
	del data_points
	del results_ax1
	del results_ax2
	del history


# plot rmse over part_train
n_days = len(data_surface)
test_days = int(PART_TEST * n_days)
axis_train_days = jnp.astype(n_days * AX_PART_TRAIN, 'int32')

fig, ax = plt.subplots(figsize=(7,5))
ax.plot(axis_train_days, [sum([h['test_rmse'][0] for h in h_ax2]) / len(h_ax2) for h_ax2 in results[0]], c='darkgreen', label="PINN")
ax.plot(axis_train_days, [sum([h['test_rmse'][0] for h in h_ax2]) / len(h_ax2) for h_ax2 in results[1]], c='darkred', linestyle='dashed', label="MSE")
ax.legend()
ax.set_ylabel("Cross-validated test RMSE (Metres)")
ax.set_xlabel("Num. days in [Train:Test] sets")
ax.set_xticks(axis_train_days[::3], [f"{train_days}:{test_days}" for train_days in axis_train_days[::3]])
ax.grid()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot all partition schemes
if PLOT_SCHEMES:
	frame_counter = 0
	for i, part_train in enumerate(AX_PART_TRAIN):
		for j, part_buffer in enumerate(AX_PART_BUFFER_FN(part_train)):
			
			test_rmse = results[0][i][j]['test_rmse'][0]
			output_name = f"{__file__.replace('.py','')}_Figure_2_{frame_counter}.png"
			
			plot_data_partition(part_buffer, part_train, title=f"Data partition in days\nTest RMSE={test_rmse:.2f}m", scale=n_days)
			plt.tight_layout()
			plt.savefig(output_name)
			plt.close()
			frame_counter += 1
			print(f"Saved \"{output_name}\"")
			print(f"[Elapsed time: {time.time()-T0:.2f}s]")
