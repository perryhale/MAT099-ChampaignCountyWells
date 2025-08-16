import sys
import time
import math
import pickle

import jax
import jax.numpy as jnp
from jax.nn import tanh
import optax
import pandas as pd
import matplotlib.pyplot as plt
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
MODEL_ACTIVATION = lambda a,b,x: jax.nn.tanh(a*x) + b*a*x*tanh(a*x) # scaled stan
MODEL_ACTIVATION_A_MIN = 1
MODEL_ACTIVATION_A_MAX = 10
MODEL_ACTIVATION_A_RES = 2
MODEL_ACTIVATION_B_MIN = 0
MODEL_ACTIVATION_B_MAX = 9
MODEL_ACTIVATION_B_RES = 2

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

def trial_fn(a, b,
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
		hidden_activation=trial_activation,
		collocation_per_input_dim=1 ###! disabling physics constraint
	)
	print(f"a={a}, b={b}, count_params(params)={count_params(params)}")
	
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
# 1)
data_points = jnp.array([(*data_wells[j], data_surface[i][0], data_surface[i][j+1]) for i in range(data_surface.shape[0]) for j in range(data_surface.shape[1]-1)])
# 2)
data_split = train_val_test_split(K0, data_points, BATCH_SIZE, part_train=PART_TRAIN, part_val=PART_VAL, part_test=PART_TEST)
# 3)
(train_x, train_y, train_steps), (val_x, val_y, val_steps), (test_x, test_y, test_steps), data_scaler = data_split
# 4)
data_scale_xytz = data_scaler.data_range_ / jnp.ones(len(data_scaler.data_range_), dtype='float32') # units (m, m, s, m)
# trace
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
		b_axis=h_cache['b_axis']
		results=h_cache['results']
		print(f"Loaded \"{H_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception:
	
	# initialise trial axes
	a_axis = jnp.linspace(MODEL_ACTIVATION_A_MIN, MODEL_ACTIVATION_A_MAX, MODEL_ACTIVATION_A_RES)
	b_axis = jnp.linspace(MODEL_ACTIVATION_B_MIN, MODEL_ACTIVATION_B_MAX, MODEL_ACTIVATION_B_RES)
	
	# iterate over trial axes
	trial_count = 0
	trial_total = len(b_axis)*len(a_axis)
	results = [[None]*len(b_axis)]*len(a_axis)
	for i,a in enumerate(a_axis):
		for j,b in enumerate(b_axis):
			results[i][j] = trial_fn(a, b,
				data_scale_xytz, k_crop,
				train_x, train_y, train_steps,
				val_x, val_y, val_steps,
				test_x, test_y, test_steps
			)
			trial_count += 1
			print(f"Completed trial {trial_count}/{trial_total}")
			print(f"[Elapsed time: {time.time()-T0:.2f}s]")
			
			# update H_CACHE
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


### plotting

import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
A, B = jnp.meshgrid(a_axis, b_axis, indexing='ij')

# -------------------------------
# Interpolate to Upscale (optional)
# -------------------------------
# Flatten grid and values for interpolation
points = jnp.stack([A.flatten(), B.flatten()], axis=-1)
values = []
for row in results:
	values.extend([h['test_loss'][0] for h in row])

# Define finer grid for interpolation
a_axis_fine = jnp.linspace(MODEL_ACTIVATION_A_MIN, MODEL_ACTIVATION_A_MAX, 200)
b_axis_fine = jnp.linspace(MODEL_ACTIVATION_B_MIN, MODEL_ACTIVATION_B_MAX, 200)
A_fine, B_fine = jnp.meshgrid(a_axis_fine, b_axis_fine, indexing='ij')
grid_fine = jnp.stack([A_fine.flatten(), B_fine.flatten()], axis=-1)

# Perform interpolation
R_interp = griddata(points=points, values=values, xi=grid_fine, method='cubic')
R_interp = R_interp.reshape(200, 200)

# -------------------------------
# Plot
# -------------------------------
plt.figure(figsize=(8, 6))
contour = plt.contourf(a_axis_fine, b_axis_fine, R_interp.T, levels=100, cmap='viridis')
plt.colorbar(contour, label='trial_fn(a, b)')
plt.xlabel("a")
plt.ylabel("b")
plt.title("Interpolated Contour Plot of trial_fn(a, b)")
plt.tight_layout()
plt.show()


print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
