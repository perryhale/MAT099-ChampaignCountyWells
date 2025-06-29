import time
import math
import pickle ###!

import jax
import jax.numpy as jnp
import optax

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from library.data import batch_generator, unit_grid2_sample_fn
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
PART_TEST = 0.2

# optimizer
ETA = 1e-4
LAM_MSE = 1.0
LAM_PHYS = 0.0
LAM_L2 = 0.1

# physical constants
SS = 1e-4
RR = 1e-7

# sampling
SAMPLE_XMIN = -2
SAMPLE_XMAX = +3
SAMPLE_XRES = 0 ###! 0 -> inherit k shape
SAMPLE_YMIN = -2
SAMPLE_YMAX = +3
SAMPLE_YRES = 0 ###! 0 -> inherit k shape
SAMPLE_TMIN = 0
SAMPLE_TMAX = 2
SAMPLE_TRES = 400


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
for i in range(0, data_surface.shape[0]-1, 1):
	for j in range(0, data_surface.shape[1]-1, 1):
		xytz = (*data_wells[j], data_surface[i][0], data_surface[i][j+1]) # xytz
		data_points.append(xytz)

data_points = jnp.array(data_points)

# partition data
n_data = len(data_points)
shuffle_idx = jax.random.permutation(K0, n_data)
data_train = data_points[shuffle_idx[:int(PART_TRAIN * n_data)]]
data_val = data_points[shuffle_idx[int(PART_TRAIN * n_data) : int((PART_TRAIN + PART_VAL) * n_data)]]
data_test = data_points[shuffle_idx[int((PART_TRAIN + PART_VAL) * n_data) : int((PART_TRAIN + PART_VAL + PART_TEST) * n_data)]]

# project to unit hypercube
data_scaler = MinMaxScaler(feature_range=(0, 1))
data_scaler.fit(data_train)

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
del n_data
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
h_param = init_dense_neural_network(K0, [3, 256, 256, 1])
h_fn = jax.vmap(lambda p,xyt: dense_neural_network(p, xyt, ha=jax.nn.relu)[0,0], in_axes=(None, 0)) # N x [0,1] x [0,1] x [0,1] -> N x [0,1]
# rr_param = init_dense_neural_network(K0, [3, 32, 32, 32, 1])
# rr_fn = h_fn

def loss_3d_ground_water_flow(params, batch_xyt):
	"""
	# loss = ||R||^2
	# R = Ss * ∂h/∂t - ∇·(K ∇h) - Rr
	# ∇h = (∂h/∂x, ∂h/∂y)
	# https://en.wikipedia.org/wiki/Groundwater_flow_equation
	# https://github.com/jax-ml/jax/issues/3022#issuecomment-2733591263
	"""
	
	h_fn_mono = lambda xyt: h_fn(params[0], xyt[jnp.newaxis, :])[0]
	h_fn_flux = lambda xyt: unit_grid2_sample_fn(k_crop, *xyt[:2]) * jax.grad(h_fn_mono)(xyt)[:2]
	
	# compute 3d groundwater flow terms
	batch_dhdt = jax.vmap(lambda xyt: jax.grad(h_fn_mono)(xyt)[2])(batch_xyt)
	batch_div_flux = jax.vmap(lambda xyt: jnp.trace(jax.jacfwd(h_fn_flux)(xyt)))(batch_xyt)
	batch_ss = SS#params[-1][0]
	batch_rr = RR#params[-1][1]
	
	# return l2 of residual
	residual = batch_ss * batch_dhdt - batch_div_flux - batch_rr
	loss = jnp.mean(residual**2)
	
	return loss

def loss_fn(params, batch_xyt, batch_z):
	
	loss_batch = LAM_MSE * loss_mse(h_fn(params[0], batch_xyt), batch_z)
	loss_phys = LAM_PHYS * loss_3d_ground_water_flow(params, batch_xyt)
	loss_reg = LAM_L2 * lp_norm(params, order=2)
	
	loss = loss_batch + loss_phys + loss_reg
	return loss

# try cache
try:
	with open(W_CACHE, 'rb') as f:
		w_cache = pickle.load(f)
		history = w_cache['history']
		params = w_cache['params']
		h_param = w_cache['params'][0]
		axis_x = w_cache['sample']['axis_x']
		axis_y = w_cache['sample']['axis_y']
		axis_t = w_cache['sample']['axis_t']
		h_sim = w_cache['sample']['h_sim']
		print(f"Loaded \"{W_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except Exception as e:
	
	# setup optimiser
	params = [h_param] ###! wrapper list to allow writer to manually add extra parameters
	opt = optax.adamw(ETA)
	opt_state = opt.init(params)
	epoch_key = K2
	history = {'batch_loss':[], 'train_loss':[], 'val_loss':[], 'test_loss':[]}
	print(f"history_keys={list(history.keys())}")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	@jax.jit
	def opt_step(opt_state_, params_, x, y):
		loss, grad = jax.value_and_grad(loss_fn)(params_, x, y)
		updates, opt_state_ = opt.update(grad, opt_state_, params_)
		params_ = optax.apply_updates(params_, updates)
		return loss, opt_state_, params_
	
	# fit model
	for i in range(EPOCHS):
		
		# setup data generators
		epoch_key = jax.random.split(epoch_key, 1)[0]
		train_generator = batch_generator(train_x, train_y, BATCH_SIZE, shuffle_key=epoch_key)
		val_generator = batch_generator(val_x, val_y, BATCH_SIZE)
		
		# iterate optimiser
		train_loss = 0.
		for j in range(train_steps):
			
			# batch loss and update
			batch_loss, opt_state, params = opt_step(opt_state, params, *next(train_generator))
			train_loss += batch_loss / train_steps
			
			# record/trace
			history['batch_loss'].append(batch_loss)
			#print(f"[Elapsed time: {time.time()-T0:.2f}s] epoch={i+1}, batch={j+1}, batch_loss={batch_loss:.4f}")
		
		# validation loss
		val_loss = 0.
		for _ in range(val_steps):
			val_loss += loss_fn(params, *next(val_generator)) / val_steps
		
		# record/trace
		history['train_loss'].append(train_loss)
		history['val_loss'].append(val_loss)
		print(f"[Elapsed time: {time.time()-T0:.2f}s] epoch={i+1}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
	
	# compute test loss
	test_generator = batch_generator(test_x, test_y, BATCH_SIZE)
	test_loss = 0.
	for _ in range(test_steps):
		test_loss += loss_fn(params, *next(test_generator)) / test_steps
	
	# record/trace
	history['test_loss'] = test_loss
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# sample surface
	axis_x = jnp.linspace(SAMPLE_XMIN, SAMPLE_XMAX, k_crop.shape[1] if (SAMPLE_XRES==0) else SAMPLE_XRES)
	axis_y = jnp.linspace(SAMPLE_YMIN, SAMPLE_YMAX, k_crop.shape[0] if (SAMPLE_YRES==0) else SAMPLE_YRES)
	axis_t = jnp.linspace(SAMPLE_TMIN, SAMPLE_TMAX, SAMPLE_TRES)
	h_sim = h_fn(params[0], jnp.stack(jnp.meshgrid(axis_t, axis_y, axis_x, indexing='ij')[::-1], axis=-1).reshape(-1, 3))
	h_sim = data_scaler.data_min_[3] + h_sim * data_scaler.data_range_[3]
	h_sim = h_sim.reshape(len(axis_t), len(axis_y), len(axis_x))
	print(h_sim.shape)
	print(h_sim.min())
	print(h_sim.max())
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# create cache
	with open(W_CACHE, 'wb') as f:
		pickle.dump(dict(
			params=params,
			history=history,
			sample=dict(
				axis_x=axis_x,
				axis_y=axis_y,
				axis_t=axis_t,
				h_sim=h_sim
			)
		), f)
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
	grid_extent=(axis_x.min(), axis_x.max(), axis_y.min(), axis_y.max()), ###!
	draw_box=(0,0,1,1),
	draw_k_in_box=True,
	cmap_contour='binary',
	axis_ticks=True,
	origin=None,
	isolines=25,
	scatter_data=data_scatter.T,
	save_path="DFRPINN_EXTRAPOLATED_SURFACE_250629.webm"
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
