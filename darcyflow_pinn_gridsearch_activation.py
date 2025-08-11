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

###! variations

# MODEL_ACTIVATION = lambda b,x: jax.nn.tanh(x) + b * x * jax.nn.tanh(x) # stan
# MODEL_ACTIVATION_B_MIN = 0
# MODEL_ACTIVATION_B_MAX = 9
# MODEL_ACTIVATION_B_RES = 8

# MODEL_ACTIVATION = lambda b,x: jax.nn.tanh(b*x) # squashed tanh
# MODEL_ACTIVATION_B_MIN = 1
# MODEL_ACTIVATION_B_MAX = 10
# MODEL_ACTIVATION_B_RES = 8

MODEL_ACTIVATION = lambda b,x: jax.nn.relu(b*x) # squashed relu
MODEL_ACTIVATION_B_MIN = 1
MODEL_ACTIVATION_B_MAX = 10
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
SAMPLE_ACTIVATION = jnp.linspace(-2, 2, 512)


### functions

def trial_fn(b,
		data_scale_xytz, k_crop, train_x, train_y, train_steps, val_x, val_y, val_steps, test_x, test_y, test_steps
	):
	
	# init model
	trial_activation = lambda x: MODEL_ACTIVATION(b, x)
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
	print(f"b={b}, count_params(params)={count_params(params)}")
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
	history['sample'] = dict(
		axis_x=axis_x,
		axis_y=axis_y,
		axis_t=axis_t,
		h_sim=h_sim
	)
	
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

# populate data_points
data_points = []
for i in range(0, data_surface.shape[0], 1):
	for j in range(0, data_surface.shape[1]-1, 1):
		xytz = (*data_wells[j], data_surface[i][0], data_surface[i][j+1]) # xytz
		data_points.append(xytz)

data_points = jnp.array(data_points)

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

# run trials
try:
	with open(H_CACHE, 'rb') as f:
		h_cache = pickle.load(f)
		data_scaler=h_cache['data_scaler']
		b_axis=h_cache['b_axis']
		results=h_cache['results']
		print(f"Loaded \"{H_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception:
	
	b_axis = jnp.linspace(MODEL_ACTIVATION_B_MIN, MODEL_ACTIVATION_B_MAX, MODEL_ACTIVATION_B_RES)
	results = [None]*len(b_axis)
	
	for i,b in enumerate(b_axis):
		
		# run trial
		history = trial_fn(b,
			data_scale_xytz, k_crop, train_x, train_y, train_steps, val_x, val_y, val_steps, test_x, test_y, test_steps
		)
		results[i] = history
		
		# cache checkpoint
		if CACHE_ENABLED:
			with open(H_CACHE, 'wb') as f:
				h_cache = dict(
					data_scaler=data_scaler,
					b_axis=b_axis,
					results=results
				)
				pickle.dump(h_cache, f)
				print(f"Saved \"{H_CACHE}\"")
				print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# extract stats
axis_test_loss = jnp.array([h['test_loss'] for h in results])
idx_best_trial = jnp.argmin(axis_test_loss)
axis_x = results[idx_best_trial]['sample']['axis_x']
axis_y = results[idx_best_trial]['sample']['axis_y']
axis_t = results[idx_best_trial]['sample']['axis_t']
h_sim = results[idx_best_trial]['sample']['h_sim']
print(f"idx_best_trial={idx_best_trial}, b_axis[idx_best_trial]={b_axis[idx_best_trial]}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot results
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(b_axis, axis_test_loss, c='green')
ax.set_xlabel("β")
ax.set_ylabel("Test loss")
ax.set_xticks(b_axis, [f"{x:.1f}" for x in b_axis])
ax.set_xlim(b_axis.min(), b_axis.max())
ax.grid()
plt.subplots_adjust(left=0.2, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot activation functions
a_fn = MODEL_ACTIVATION
b_cols = plt.cm.Dark2(jnp.linspace(0, 1, MODEL_ACTIVATION_B_RES))
bs = b_axis
xs = SAMPLE_ACTIVATION
bys = [a_fn(b, xs) for b in bs]
bgys = [jax.vmap(jax.grad(lambda x: a_fn(b, x)))(xs) for b in bs]

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
for b, ys, gys, c in zip(bs, bys, bgys, b_cols):
	ax0.plot(xs, ys, c=c, label=f"β = {b:.1f}")
	ax1.plot(xs, gys, c=c, label=f"β = {b:.1f}")

#ax0.axhline(0, linestyle='dashed', c='black') ###! disable for relu
#ax0.axvline(0, linestyle='dashed', c='black')
ax0.set_xlim(SAMPLE_ACTIVATION.min(), SAMPLE_ACTIVATION.max())
ax0.set_ylabel("a(x)")
ax0.set_xlabel("x")
ax0.grid()
ax0.legend()

#ax1.axhline(0, linestyle='dashed', c='black')
#ax1.axvline(0, linestyle='dashed', c='black')
ax1.set_xlim(SAMPLE_ACTIVATION.min(), SAMPLE_ACTIVATION.max())
ax1.set_ylabel("δa(x) / δx")
ax1.set_xlabel("x")
ax1.grid()
ax1.legend()

plt.tight_layout()
plt.show()

# plot surfaces
fig, axis = plt.subplots(figsize=(20,3), nrows=1, ncols=MODEL_ACTIVATION_B_RES)
for ax, b, h in zip(axis, b_axis, results):
	ax_contour = ax.contour(h['sample']['h_sim'][50], levels=10, cmap='binary_r', extent=(0,1,0,1))
	ax_clabel = ax.clabel(ax_contour, inline=True, fontsize=8, colors='red')
	ax.grid()
	ax.set_xticks([],[])
	ax.set_yticks([],[])
	ax.set_title(f"β={b:.1f}\ntest_loss={h['test_loss'][0]:.4f}", fontsize=11)

plt.tight_layout()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# animate best surface
data_scatter = (data_wells - data_scaler.data_min_[:2]) / data_scaler.data_range_[:2]
animate_hydrology(
	h_sim,
	k=k_crop,
	grid_extent=(axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()),
	cmap_contour='Blues_r',
	axis_ticks=True,
	origin=None,
	isolines=10,
	scatter_data=data_scatter.T,
	title_fn=lambda t: f"t={axis_t[t]:.2f}",
	clabel_fmt='%d',
	save_path="darcyflow_pinn_animation.mp4"
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
