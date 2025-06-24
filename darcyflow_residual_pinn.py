import time
import math

import jax
import jax.numpy as jnp
import optax

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from library.data import batch_generator
from library.models.nn import *
from library.visualize import animate_hydrology, plot_surface3d


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
I_CACHE = 'data/processed/data_interpolated.npz'
S_CACHE = 'data/processed/data_surface.csv'

# RNG setup
RNG_SEED = 999
K0, K1, K2 = jax.random.split(jax.random.key(RNG_SEED), 3)

# data partitions
EPOCHS = 15
BATCH_SIZE = 64
PART_TRAIN = 0.75
PART_VAL = 0.05
PART_TEST = 0.20

# optimizer
ETA = 1e-3
LAM_MSE = 1.0
LAM_PHYS = 1.0
LAM_L2 = 0.25

# physical constants
SS = 1e-1
RR = 1e-7

# plotting
VIDEO_FRAME_SKIP = 0
VIDEO_SAVE = False


### main

# load cache
with jnp.load(I_CACHE) as data_interpolated:
	k_crop = data_interpolated['k_crop']
	data_wells = data_interpolated['data_wells']
	print("Loaded cache")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

data_surface = pd.read_csv(S_CACHE).to_numpy()

# populate xyt->z
data_points = []
for i in range(0, data_surface.shape[0]-1, 1):
	for j in range(0, data_surface.shape[1]-1, 1):
		xytz = (*data_wells[j], i, data_surface[i][j+1]) # xytz
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
del data_surface
del data_wells
del data_train
del data_val
del data_test

# trace
print(f"Train: x~{train_x.shape}, y~{train_y.shape}, steps={train_steps}")
print(f"Val: x~{val_x.shape}, y~{val_y.shape}, steps={val_steps}")
print(f"Test: x~{test_x.shape}, y~{test_y.shape}, steps={test_steps}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# initialise model
h_param = init_dense_neural_network(K0, [3, 32, 32, 32, 1])
rr_param = init_dense_neural_network(K1, [3, 32, 32, 32, 1])
h_fn = lambda p,xyt: jax.nn.sigmoid(dense_neural_network(p, xyt, a=jax.nn.tanh)) # [0,1]x[0,1]x[0,1] -> [0,1]
k_fn = lambda xy: jax.lax.dynamic_slice(k_crop, (jnp.floor(xy[1] * k_crop.shape[0]-1).astype(jnp.int32), jnp.floor(xy[0] * k_crop.shape[1]-1).astype(jnp.int32)), (1, 1))[0, 0] # [0,1]x[0,1] -> R
#rr_fn = lambda p,xyt: RR # [0,1]x[0,1]x[0,1] -> R
rr_fn = lambda p,xyt: dense_neural_network(p, xyt, a=jax.nn.tanh)#RR # [0,1]x[0,1]x[0,1] -> R

def loss_3dgwf(params, batch_xyt):
	
	# setup
	h_fn_ = lambda xyt_: h_fn(params[0], xyt_)[0]
	flux_fn = lambda xyt: k_fn(xyt[:2]) * jax.grad(h_fn_)(xyt)[:2] # ∇h = (∂h/∂x, ∂h/∂y)
	
	# compute 3d groundwater flow terms
	batch_dhdt = jax.vmap(lambda xyt: jax.grad(h_fn_)(xyt)[2])(batch_xyt)
	batch_div_flux = jax.vmap(lambda xyt: jnp.trace(jax.jacfwd(flux_fn)(xyt)))(batch_xyt)
	batch_rr = jax.vmap(rr_fn, in_axes=(None, 0))(params[1], batch_xyt)
	
	# return l2 of residual
	residual = SS * batch_dhdt - batch_div_flux - batch_rr # Ss * ∂h/∂t - ∇·(K ∇h) - Rr
	loss = jnp.mean(residual**2)
	
	return loss

def loss_fn(params, batch_xyt, batch_z):
	batch_zh = jax.vmap(lambda xyt: h_fn(params[0], xyt))(batch_xyt)
	loss = LAM_MSE*loss_mse(batch_zh, batch_z) + LAM_PHYS*loss_3dgwf(params, batch_xyt) + LAM_L2*lp_norm(params, order=2)
	return loss

# setup optimiser
params = [h_param, rr_param]
optim = optax.adamw(ETA)
state = optim.init(params)
history = {'batch_l2':[], 'batch_loss':[], 'train_loss':[], 'val_loss':[], 'test_loss':[]}
print(f"history_keys={list(history.keys())}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

@jax.jit
def optimizer_step(s, p, x, y):
	loss, grad = jax.value_and_grad(loss_fn)(p, x, y)
	updates, next_state = optim.update(grad, s, p)
	next_params = optax.apply_updates(p, updates)
	return loss, next_state, next_params

# fit model
batch_key = K2
for i in range(EPOCHS):
	
	# setup data generators
	batch_key = jax.random.split(batch_key, 1)[0]
	train_generator = batch_generator(train_x, train_y, BATCH_SIZE, shuffle_key=batch_key)
	val_generator = batch_generator(val_x, val_y, BATCH_SIZE)
	
	# iterate optimiser
	train_loss = 0.
	for j in range(train_steps):
		
		# batch loss and update
		batch_loss, state, params = optimizer_step(state, params, *next(train_generator))
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

# record test statistics
history['test_loss'] = test_loss

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






import sys;sys.exit()





### Autoregressive simulation

# simulate
#h_init = jnp.ones(k_crop.shape)
#h_init = jnp.array([[jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y) for x in jnp.linspace(0, 1, k_crop.shape[1])] for y in jnp.linspace(0, 1, k_crop.shape[0])])
state = test_x[0]
h_sim = [state]
for _ in tqdm(range(len(test_x))):
	state = pinn_model(params[:-1], state.reshape((1, grid_flat_size, ))).reshape((grid_shape))
	#state = apply_edge_boundary_conditions(state)
	h_sim.append(state)
h_sim = jnp.array(h_sim)
h_sim = h_sim*data_train_range+data_train_min
print(f"Simulation completed.")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# compute/plot simulation statistics
h_sim_mean = [h.mean() for h in h_sim[1:]]
h_sim_var = [h.var() for h in h_sim[1:]]
h_sim_rmse = [loss_mse(yh, y)**0.5 for yh,y in zip(h_sim[1:], test_y)]
plt.plot(h_sim_mean, label="Mean")
plt.plot(h_sim_var, label="Var")
plt.plot(h_sim_rmse, label="RMSE")
plt.legend()
plt.grid()
plt.xlabel("Time (days)")
plt.ylabel("Height (metres)")
plt.show()

# animate simulation
animate_hydrology(
	h_sim,
	k=k_crop,
	axis_ticks=True,
	frame_skip=VIDEO_FRAME_SKIP,
	save_path=__file__.replace('.py','.mp4') if VIDEO_SAVE else None
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot 3d surface
fig, axis = plot_surface3d(grid_x, grid_y, h_sim[-1], k=k_crop)
plt.tight_layout()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
