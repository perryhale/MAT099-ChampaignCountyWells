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

from library.data.pipeline import batch_generator, train_val_test_split
from library.models.nn import get_3d_groundwater_flow_model, sample_3d_model
from library.models.util import fit, count_params
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
EPOCHS = 2
BATCH_SIZE = 64
PART_TRAIN = 0.75
PART_VAL = 0.05
PART_TEST = 0.20

# model
MODEL_LAYERS = [3, 256, 256, 1]
MODEL_ACTIVATION = lambda a,x: jax.nn.relu(a*x) # scaled relu
#MODEL_ACTIVATION = lambda a,x: jax.nn.tanh(a*x) # scaled tanh
#MODEL_ACTIVATION = lambda a,x: jax.nn.tanh(a*x) + a*x*jax.nn.tanh(a*x) # scaled stan
MODEL_ACTIVATION_A_MIN = 1
MODEL_ACTIVATION_A_MAX = 10
MODEL_ACTIVATION_A_RES = 8

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

def trial_fn(a,
		data_scale_xytz, k_crop, train_x, train_y, train_steps, val_x, val_y, val_steps, test_x, test_y, test_steps
	):
	
	# init model
	trial_activation = lambda x: MODEL_ACTIVATION(a, x)
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
	print(f"a={a}, count_params(params)={count_params(params)}")
	
	# fit model
	params, history = fit(
		K2,
		params,
		loss_fn,
		(train_x, train_y, train_steps),
		val_data=(val_x, val_y, val_steps),
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
		opt=OPT(OPT_ETA),
		start_time=T0
	)
	print(f"ss={float(params[-1][0])}, rr={float(params[-1][1])}")
	
	# test model
	test_generator = batch_generator(test_x, test_y, BATCH_SIZE)
	test_loss = 0.
	for _ in range(test_steps):
		test_loss += loss_fn(params, *next(test_generator)) / test_steps
	print(f"test_loss={test_loss:.4f}")
	
	# sample model
	axis_x = jnp.linspace(SAMPLE_3D_XMIN, SAMPLE_3D_XMAX, k_crop.shape[1] if (SAMPLE_3D_XRES==0) else SAMPLE_3D_XRES)
	axis_y = jnp.linspace(SAMPLE_3D_YMIN, SAMPLE_3D_YMAX, k_crop.shape[0] if (SAMPLE_3D_YRES==0) else SAMPLE_3D_YRES)
	axis_t = jnp.linspace(SAMPLE_3D_TMIN, SAMPLE_3D_TMAX, SAMPLE_3D_TRES)
	h_sim = sample_3d_model(h_fn, params[0], axis_t, axis_y, axis_x, batch_size=None)
	h_sim = data_scaler.data_min_[3] + h_sim * data_scaler.data_range_[3]
	print(f"h_sim.shape={h_sim.shape}")
	
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

# prepare data
data_points = jnp.array([(*data_wells[j], data_surface[i][0], data_surface[i][j+1]) for i in range(data_surface.shape[0]) for j in range(data_surface.shape[1]-1)])
data_split = train_val_test_split(K0, data_points, BATCH_SIZE, part_train=PART_TRAIN, part_val=PART_VAL, part_test=PART_TEST)
(train_x, train_y, train_steps), (val_x, val_y, val_steps), (test_x, test_y, test_steps), data_scaler = data_split
data_scale_xytz = data_scaler.data_range_ / jnp.ones(len(data_scaler.data_range_), dtype='float32') # units (m, m, s, m)
print(f"Train: x~{train_x.shape}, y~{train_y.shape}, steps={train_steps}")
print(f"Val: x~{val_x.shape}, y~{val_y.shape}, steps={val_steps}")
print(f"Test: x~{test_x.shape}, y~{test_y.shape}, steps={test_steps}")
print(f"Scale: [{', '.join([f'{float(scale.item()):.1f}{unit}' for scale, unit in zip(data_scale_xytz, 'mmsm')])}]")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# run trials
try:
	with open(H_CACHE, 'rb') as f:
		h_cache = pickle.load(f)
		data_scaler=h_cache['data_scaler']
		a_axis=h_cache['a_axis']
		results=h_cache['results']
		print(f"Loaded \"{H_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception:
	
	a_axis = jnp.linspace(MODEL_ACTIVATION_A_MIN, MODEL_ACTIVATION_A_MAX, MODEL_ACTIVATION_A_RES)
	results = [None]*len(a_axis)
	
	for i,a in enumerate(a_axis):
		
		# run trial
		history = trial_fn(a,
			data_scale_xytz, k_crop, train_x, train_y, train_steps, val_x, val_y, val_steps, test_x, test_y, test_steps
		)
		results[i] = history
		
		# cache checkpoint
		if CACHE_ENABLED:
			with open(H_CACHE, 'wb') as f:
				h_cache = dict(
					data_scaler=data_scaler,
					a_axis=a_axis,
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
print(f"idx_best_trial={idx_best_trial}, a_axis[idx_best_trial]={a_axis[idx_best_trial]}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot results
fig, ax = plt.subplots(figsize=(4,3))
ax.plot(a_axis, axis_test_loss, c='green')
ax.set_xlabel("α")
ax.set_ylabel("Test loss")
ax.set_xticks(a_axis, [f"{x:.1f}" for x in a_axis])
ax.set_xlim(a_axis.min(), a_axis.max())
ax.grid()
plt.subplots_adjust(left=0.2, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot activation functions
a_fn = MODEL_ACTIVATION
a_cols = plt.cm.Dark2(jnp.linspace(0, 1, MODEL_ACTIVATION_A_RES))
xs = SAMPLE_ACTIVATION
ays = [a_fn(a, xs) for a in a_axis]
agys = [jax.vmap(jax.grad(lambda x: a_fn(a, x)))(xs) for a in a_axis]

fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
for a, ys, gys, c in zip(a_axis, ays, agys, a_cols):
	ax0.plot(xs, ys, c=c, label=f"α={a:.1f}")
	ax1.plot(xs, gys, c=c, label=f"α={a:.1f}")

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
fig, axis = plt.subplots(figsize=(20,3), nrows=1, ncols=MODEL_ACTIVATION_A_RES)
for ax, a, h in zip(axis, a_axis, results):
	ax_contour = ax.contour(h['sample']['h_sim'][50], levels=10, cmap='binary_r', extent=(0,1,0,1))
	ax_clabel = ax.clabel(ax_contour, inline=True, fontsize=8, colors='red')
	ax.grid()
	ax.set_xticks([],[])
	ax.set_yticks([],[])
	ax.set_title(f"α={a:.1f}\ntest_loss={h['test_loss'][0]:.4f}", fontsize=11)

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
