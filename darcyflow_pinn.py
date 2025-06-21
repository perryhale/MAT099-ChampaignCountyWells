import time
import math
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from library.data import batch_generator
from library.models.fdm import darcyflow_fdm_periodic, cfl_value
from library.models.nn import *
from library.visualize import animate_hydrology, plot_surface3d


### Setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
I_CACHE = 'data/processed/data_interpolated.npz'

# RNG setup
RNG_SEED = 999
K0, K1 = jax.random.split(jax.random.key(RNG_SEED))

# FDM disretization and units
DX = 1000
DY = DX
DT = 24
#K_SCALE = 1 # cm/hr
K_SCALE = 100 # m/hr
#K_SCALE = 36e4**-1 # m/s
#K_SCALE = 24e-5 # km/day

# data partitions
EPOCHS = 10
BATCH_SIZE = 32
PART_TRAIN = 0.75
PART_VAL = 0.05
PART_TEST = 0.20

# optimizer
ETA = 1e-3
REG_PHYS = 0.1
REG_L2 = 0.025

# plotting
VIDEO_FRAME_SKIP = 0
VIDEO_SAVE = False


### Prepare data

# load cache
with jnp.load(I_CACHE) as data_interpolated:
	k_crop = data_interpolated['k_crop']
	h_time = data_interpolated['h_time']
	grid_x = data_interpolated['grid_x']
	grid_y = data_interpolated['grid_y']
	print("Loaded cache")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# rescale K
k_crop = k_crop * K_SCALE
print(f"K mean: {k_crop.mean()}")
print(f"K var: {k_crop.var()}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# measure grid
grid_shape = k_crop.shape
grid_flat_size = jnp.prod(jnp.array(grid_shape))
print(f"grid_shape={grid_shape}")
print(f"grid_flat_size={grid_flat_size}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

###! truncate data
h_time = h_time[4_042:]
print(f"Truncated length: {h_time.shape[0]}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# partition data
data_train = h_time[:int(PART_TRAIN*len(h_time))]
data_val = h_time[int(PART_TRAIN*len(h_time)) : int((PART_TRAIN+PART_VAL)*len(h_time))]
data_test = h_time[int((PART_TRAIN+PART_VAL)*len(h_time)) : int((PART_TRAIN+PART_VAL+PART_TEST)*len(h_time))]

# normalise data
data_train_min = data_train.min()
data_train_range = jnp.ptp(data_train)
data_train = (data_train - data_train_min) / data_train_range
data_val = (data_val - data_train_min) / data_train_range
data_test = (data_test - data_train_min) / data_train_range

# convert to time-series
train_x, train_y = data_train[:-1], data_train[1:]
val_x, val_y = data_val[:-1], data_val[1:]
test_x, test_y = data_test[:-1], data_test[1:]

# determine batch counts
train_steps = math.ceil(train_x.shape[0] / BATCH_SIZE)
val_steps = math.ceil(val_x.shape[0] / BATCH_SIZE)
test_steps = math.ceil(test_x.shape[0] / BATCH_SIZE)

# memory cleanup
del h_time; del data_train; del data_val; del data_test

print(f"Train: x~{train_x.shape} y~{train_y.shape}, min={jnp.min(jnp.minimum(train_x, train_y)):.2f}, max={jnp.max(jnp.maximum(train_x, train_y)):.2f}, steps={train_steps}")
print(f"Val: x~{val_x.shape} y~{val_y.shape}, min={jnp.min(jnp.minimum(val_x, val_y)):.2f}, max={jnp.max(jnp.maximum(val_x, val_y)):.2f}, steps={val_steps}")
print(f"Test: x~{test_x.shape} y~{test_y.shape}, min={jnp.min(jnp.minimum(test_x, test_y)):.2f}, max={jnp.max(jnp.maximum(test_x, test_y)):.2f}, steps={test_steps}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")


### Fit PINN

# initialise model
params = [
	*init_dense_neural_network(K0, [grid_flat_size, 1_500//4, 750//4, 1_500//4, grid_flat_size]),
	[0.1, 0.0]
]
#pinn_model = jax.vmap(lambda p,x: jax.nn.sigmoid(dense_neural_network(p, x, a=jax.nn.tanh)), in_axes=(None, 0))
pinn_model = jax.vmap(lambda p,x: dense_neural_network(p, x), in_axes=(None, 0))
fdm_model = jax.vmap(darcyflow_fdm_periodic, in_axes=[0]+6*[None])

# define loss function
# type: (List[Tuple[jnp.array]], jnp.array, jnp.array) -> float
def loss_fn(p, x, y):
	yh = pinn_model(p[:-1], x.reshape((x.shape[0], -1, ))).reshape(x.shape)
	yh_fdm = fdm_model(x, k_crop, DT, DX, DY, *p[-1])
	loss = loss_mse(yh, y) + REG_PHYS*loss_mse(yh, yh_fdm) + REG_L2*lp_norm(p[:-1], order=2)
	return loss

print(f"DBG param length: #{len(params)}")
print(f"DBG e.g. loss: {loss_fn(params, val_x, val_y)}")
print(f"Parameters: #{count_params(params)}")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# setup optimzer
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
batch_key = K1
for i in range(EPOCHS):
	
	# setup data generators
	batch_key = jax.random.split(batch_key, 1)[0]
	train_generator = batch_generator(train_x, train_y, BATCH_SIZE, shuffle_key=batch_key)
	val_generator = batch_generator(val_x, val_y, BATCH_SIZE)
	
	# compute train loss and gradient
	train_loss = 0.
	for j in range(train_steps):
	
		# cfl stability check
		cfl = cfl_value(k_crop, DT, DX, DY, params[-1][0])
		if cfl >= 0.25:
			print(f"WARN: Proceeding with unstable simulation. CFL condition (CFL<0.25) not satisfied (CFL={cfl:.3f}), reduce dt or increase dx.")
		
		# step optimzer
		batch_loss, state, params = optimizer_step(state, params, *next(train_generator))
		train_loss += batch_loss * train_steps**-1
		
		# record batch statistics
		history['batch_loss'].append(batch_loss)
		history['batch_l2'].append(lp_norm(params[:-1], order=2))
		#print(f"[Elapsed time: {time.time()-T0:.2f}s] epoch={i+1}, batch={j+1}, batch_loss={batch_loss:.4f}")
	
	# compute val loss
	val_loss = 0.
	for _ in range(val_steps):
		val_loss += loss_fn(params, *next(val_generator)) * val_steps**-1
		
	# record epoch statistics
	history['train_loss'].append(train_loss)
	history['val_loss'].append(val_loss)
	print(f"[Elapsed time: {time.time()-T0:.2f}s] epoch={i+1}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# compute test loss
test_generator = batch_generator(test_x, test_y, BATCH_SIZE)
test_loss = 0.
for _ in range(test_steps):
	test_loss += loss_fn(params, *next(test_generator)) * test_steps**-1

# record test statistics
history['test_loss'] = test_loss

# plot history
plt.plot(range(train_steps*EPOCHS), history['batch_loss'], label="Batch", c='purple')
plt.plot(range(train_steps, train_steps*(EPOCHS+1), train_steps), history['train_loss'], label="Train", c='C0')
plt.plot(range(train_steps, train_steps*(EPOCHS+1), train_steps), history['val_loss'], label="Val", c='red')
plt.scatter([train_steps*EPOCHS], history['test_loss'], label="Test", c='green', marker='x')
plt.legend()
plt.xlabel("Step")
plt.ylabel("Loss")
plt.text(
	0.99*train_steps*EPOCHS,
	min(history['train_loss']) + 0.08*(max(history['train_loss']) - min(history['train_loss'])),
	f"ss={params[-1][0]:.6f}\n rr={params[-1][1]:.6f}",
	c='r', ha='right'
)
plt.grid()
plt.show()


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
