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
MODEL_ACTIVATION_A_MAX = 5
MODEL_ACTIVATION_A_RES = 16
MODEL_ACTIVATION_B_MIN = 0
MODEL_ACTIVATION_B_MAX = 4
MODEL_ACTIVATION_B_RES = 16

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
	h_fn = jax.jit(h_fn)
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
	
	# evaluate model
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
	# results = [[None]*len(b_axis)]*len(a_axis) ###! repeats the same list reference across all rows, so every column points to the last
	results = [[None for _ in range(len(b_axis))] for _ in range(len(a_axis))] ###! jnp.array does not have dtype object, so must use linked list
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


# extract values
values = (jnp.array([[h['test_loss'][0] for h in row] for row in results]) ** 0.5 ) * 100 # cm
values_sorted_index = jnp.array([jnp.unravel_index(flat_index, values.shape) for flat_index in jnp.argsort(values, axis=None)])
min_i, min_j = values_sorted_index[0] # jnp.unravel_index(values.argmin(), values.shape)

surfaces = jnp.array([[h['sample']['h_sim'] for h in row] for row in results])
axis_x = results[0][0]['sample']['axis_x']
axis_y = results[0][0]['sample']['axis_y']
axis_t = results[0][0]['sample']['axis_t']

###! clip values
# values = jnp.minimum(30.0, values)

###! crop AOI
# values = (jnp.minimum(0.025, jnp.array([[h['test_loss'][0] for h in row] for row in results])[:6,:6]) ** 0.5 ) * 100 # cm
# a_axis = a_axis[:6]
# b_axis = b_axis[:6]
# min_i, min_j = jnp.unravel_index(values.argmin(), values.shape)

###! linear interpolation (upscale)
# from scipy.interpolate import griddata
# target_res = 32
# points = jnp.stack(jnp.meshgrid(a_axis, b_axis), axis=-1).reshape(-1,2)
# a_axis_fine = jnp.linspace(a_axis.min(), a_axis.max(), target_res)
# b_axis_fine = jnp.linspace(b_axis.min(), b_axis.max(), target_res)
# grid_fine = jnp.stack(jnp.meshgrid(a_axis_fine, b_axis_fine, indexing='ij'), axis=-1).reshape(-1,2)
# values = griddata(points=points, values=values.reshape(-1), xi=grid_fine, method='linear').reshape(target_res, target_res).T
# a_axis = a_axis_fine
# b_axis = b_axis_fine
# values_sorted_index = jnp.array([jnp.unravel_index(flat_index, values.shape) for flat_index in jnp.argsort(values, axis=None)])
# min_i, min_j = values_sorted_index[0] # jnp.unravel_index(values.argmin(), values.shape)


# plot loss contour

###! logarithmically sampled colour map
from matplotlib.colors import ListedColormap
levels = 1024
log_rainbow = ListedColormap(plt.cm.rainbow(1 - jnp.logspace(0, -7, num=levels, base=10)))

plt.figure(figsize=(7, 6))
contour = plt.contourf(a_axis, b_axis, values.T, levels=levels, cmap=log_rainbow)
cbar = plt.colorbar(contour, label="Test RMSE in Centimetres", shrink=0.9, pad=0.05)
cbar.minorticks_on()
plt.scatter([a_axis[min_i]], [b_axis[min_j]], marker="*", s=256, c='gold')
plt.text(a_axis[min_i]-0.1, b_axis[min_j]+0.1, f"{values[min_i][min_j]:.2f}cm", fontsize=8, c='red', horizontalalignment='right', verticalalignment='bottom')

###! label top values
for i, (index_i, index_j) in enumerate(values_sorted_index[:128]):
	plt.text(a_axis[index_i], b_axis[index_j], f"[{i+1}]\n{values[index_i][index_j]:.1f}", fontsize=6, c='red', horizontalalignment='center', verticalalignment='center')

plt.xlabel("α")
plt.ylabel("β")
plt.gca().set_aspect('equal', 'box')
plt.tight_layout()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")


# plot surfaces
sub_step_i = values.shape[0]//2
sub_step_j = values.shape[1]//2
sub_index_t = SAMPLE_3D_TRES//2
sub_values = values[::sub_step_i,::sub_step_j]
sub_surfaces = surfaces[::sub_step_i,::sub_step_j]
sub_a_axis = a_axis[::sub_step_i]
sub_b_axis = b_axis[::sub_step_j]

nrows, ncols = sub_values.shape
fig_size = 2 * max(nrows, ncols)
fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_size, fig_size))

for i, row in enumerate(axis):
	for j, ax in enumerate(row):
		ax_contour = ax.contour(sub_surfaces[i][j][sub_index_t], levels=10, cmap='binary_r', extent=(0,1,0,1))
		ax_clabel = ax.clabel(ax_contour, inline=True, fontsize=8, colors='red')
		ax.grid()
		ax.set_xticks([])
		ax.set_yticks([])
		ax.set_title(f"α={sub_a_axis[i]:.1f}\nβ={sub_b_axis[j]:.1f}\nRMSE={sub_values[i][j]:.2f}cm", fontsize=8)
		ax.set_aspect('equal')

plt.tight_layout()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# animate optimal surface
data_scatter = (data_wells - data_scaler.data_min_[:2]) / data_scaler.data_range_[:2]
animate_hydrology(
	surfaces[min_i][min_j],
	k=k_crop,
	grid_extent=(
		axis_x.min(),
		axis_x.max(),
		axis_y.min(),
		axis_y.max()
	),
	cmap_contour='Blues_r',
	axis_ticks=True,
	origin=None,
	isolines=10,
	scatter_data=data_scatter.T,
	title_fn=lambda t: f"t={axis_t[t]:.2f}",
	clabel_fmt='%d',
#	save_path="darcyflow_pinn_gridsearch_activation_stan_Fig2_animation.mp4"
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
