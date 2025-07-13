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

from library.data import batch_generator
from library.models.nn import *
from library.visual import *


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
I_CACHE = 'cache/data_interpolated.npz'
S_CACHE = 'cache/data_surface.csv'
W_CACHE = 'cache/df_rpinn.pkl'

# RNG setup
RNG_SEED = 999
K0, K1, K2 = jax.random.split(jax.random.key(RNG_SEED), 3)

# data partitions
EPOCHS = 2
BATCH_SIZE = 64
PART_TRAIN = 0.75
PART_VAL = 0.05
PART_TEST = 0.20

# optimizer
ETA = 1e-4
LAM_MSE = 1.0
LAM_PHYS = 1.0
LAM_L2 = 0.0

# physical constants (default)
SS = 1e-5
RR = 0.0

# sampling
SAMPLE_XMIN = 0#-2
SAMPLE_XMAX = 1#+3
SAMPLE_XRES = 0 ###! 0 -> inherit k shape
SAMPLE_YMIN = 0#-2
SAMPLE_YMAX = 1#+3
SAMPLE_YRES = 0 ###! 0 -> inherit k shape
SAMPLE_TMIN = 0
SAMPLE_TMAX = 2.6
SAMPLE_TRES = 260
SAMPLE_BATCH = False


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

###! 
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
del data_points
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

# try cache
try:
	with open(W_CACHE, 'rb') as f:
		w_cache = pickle.load(f)
		data_scaler = w_cache['data_scaler']
		history = w_cache['history']
		params = w_cache['params']
		axis_x = w_cache['sample']['axis_x']
		axis_y = w_cache['sample']['axis_y']
		axis_t = w_cache['sample']['axis_t']
		h_sim = w_cache['sample']['h_sim']
		print(f"Loaded \"{W_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception as e:
	
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
	test_generator = batch_generator(test_x, test_y, BATCH_SIZE)
	test_loss = 0.
	for _ in range(test_steps):
		test_loss += loss_fn(params, *next(test_generator)) / test_steps
	
	history['test_loss'] = [test_loss]
	print(f"test_loss={test_loss:.4f}")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# sample surface
	axis_x = jnp.linspace(SAMPLE_XMIN, SAMPLE_XMAX, k_crop.shape[1] if (SAMPLE_XRES==0) else SAMPLE_XRES)
	axis_y = jnp.linspace(SAMPLE_YMIN, SAMPLE_YMAX, k_crop.shape[0] if (SAMPLE_YRES==0) else SAMPLE_YRES)
	axis_t = jnp.linspace(SAMPLE_TMIN, SAMPLE_TMAX, SAMPLE_TRES)
	h_sim = sample_3d_model(h_fn, params[0], axis_t, axis_y, axis_x, batch_size=None)
	h_sim = data_scaler.data_min_[3] + h_sim * data_scaler.data_range_[3]
	print(f"h_sim.shape={h_sim.shape}")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# create cache
	with open(W_CACHE, 'wb') as f:
		pickle.dump(dict(data_scaler=data_scaler, history=history, params=params, sample=dict(axis_x=axis_x, axis_y=axis_y, axis_t=axis_t, h_sim=h_sim)), f)
	
	print(f"Saved \"{W_CACHE}\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot history
plt.plot(range(train_steps*EPOCHS), history['batch_loss'], label="Batch", c='purple')
plt.plot(range(train_steps, train_steps*(EPOCHS+1), train_steps), history['train_loss'], label="Train", c='C0')
plt.plot(range(train_steps, train_steps*(EPOCHS+1), train_steps), history['val_loss'], label="Val", c='red')
plt.scatter([train_steps*EPOCHS], history['test_loss'], label="Test", c='green', marker='x')
plt.legend()
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot surface
fig, ax = plt.subplots(figsize=(5, 5))
ax_contour = ax.contour(h_sim[50], levels=10, cmap='binary_r', extent=(0,1,0,1))
ax_clabel = ax.clabel(ax_contour, inline=True, fontsize=8, colors='red')
ax.grid()
ax.set_xticks([],[])
ax.set_yticks([],[])
plt.tight_layout()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

fig, ax = plot_surface3d(*jnp.meshgrid(axis_x, axis_y), h_sim[50], sfc_cmap='jet', cnt_cmap='binary_r', xlabel='x', ylabel='y', zlabel='z')
fig_loop_ctrl = [False, 0]
def fig_loop_break_fn():
	fig_loop_ctrl[0] = not fig_loop_ctrl[0]
fig.canvas.mpl_connect('close_event', lambda event: fig_loop_break_fn())
while True:
	fig_loop_ctrl[1] += 45
	ax.view_init(elev=25, azim=(270+fig_loop_ctrl[1])%360)
	plt.draw()
	plt.pause(1.0)
	if fig_loop_ctrl[0]:
		plt.close(fig)
		break
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

data_scatter = (data_wells - data_scaler.data_min_[:2]) / data_scaler.data_range_[:2]
animate_hydrology(
	h_sim,
	k=k_crop,
	grid_extent=(axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()),
	draw_box=(0,0,1,1),
	draw_k_in_box=True,
	cmap_contour='Blues_r',#'binary',
	axis_ticks=True,
	origin=None,
	isolines=10,
	scatter_data=data_scatter.T,
	title_fn=lambda t: f"t={axis_t[t]:.2f}",
	clabel_fmt='%d',
#	save_path="Figure_4.mp4"
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
