import time
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from library.models import solve_darcy_fdm


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
I_CACHE = 'data/processed/data_interpolated.npz'

# RNG setup
RNG_SEED = 999
K0 = jax.random.key(RNG_SEED)

# grid scale
DX = 100_000 # cm
DY = DX # cm
DT = 24 # hr

# optimizer
EPOCHS = 1
BATCH_SIZE = 16
ETA = 1e-6
RHO = 25e-2


### functions

# type: (np.ndarray, np.ndarray, int, bool) ~> Tuple[np.ndarray, np.ndarray]
def batch_generator(data_x, data_y, batch_size, shuffle_key=None):
	
	# assertions
	assert (len(data_x)==len(data_y))
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else range(n_samples)
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			batch_y = data_y[batch_idx]
			yield batch_x, batch_y

### main

# load cache
with jnp.load(I_CACHE) as data_interpolated:
	k = data_interpolated['k_crop']
	h_time = data_interpolated['h_time']
	print("Loaded cache")
	print(k.shape)
	print(h_time.shape)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# # rescale K
# k_scaled = k * 24e-5 # km/day
# print(k_scaled.mean())
# print(k_scaled.var())

# truncate+partition data
data_x = h_time[4_000:][:-1]
data_y = h_time[4_000:][1:]
print(data_x.shape)
print(data_y.shape)

# define model
params = jnp.array((0.1, 0.1))
model = jax.vmap(lambda params,x: solve_darcy_fdm(x, k, DT, DX, DY, params[0], params[1]), in_axes=(None, 0))
loss_fn = lambda params,x,y: jnp.mean(jnp.pow(y - model(params, x), 2))
print(params)

# setup optimzer
optim = optax.sgd(ETA, momentum=RHO)
state = optim.init(params)
epoch_key = K0
n_batch = int(data_x.shape[0] / BATCH_SIZE)
history = {'loss':[], 'params':[]}
print(n_batch)

@jax.jit
def optimizer_step(state, params, x, y):
	loss, grad = jax.value_and_grad(loss_fn)(params, x, y)
	updates, state = optim.update(grad, state, params)
	next_params = optax.apply_updates(params, updates)
	return loss, state, next_params

@jax.jit
def cfl_value(k, dt, dx, dy, ss):
	return jnp.max(k) * dt * (1 / dx**2 + 1 / dy**2) / ss

# fit model
for i in range(EPOCHS):
	
	# setup data generator
	epoch_key = jax.random.split(epoch_key, 1)[0]
	data_generator = batch_generator(data_x, data_y, BATCH_SIZE, shuffle_key=epoch_key)
	
	# iterate optimizer
	for _ in range(n_batch):
		
		# compute loss and gradient
		batch_x, batch_y = next(data_generator)
		batch_loss, state, params = optimizer_step(state, params, batch_x, batch_y)
		
		# record
		history['loss'].append(batch_loss)
		history['params'].append(params)
		print(f"[Elapsed time: {time.time()-T0:.2f}s] epoch={i}, loss={batch_loss}, params={params}")
		
		# stability check (Courant–Friedrichs–Lewy)
		if cfl_value(k, DT, DX, DY, params[0]) >= 0.25:
			print(f"WARN: Proceeding with unstable simulation. CFL condition (CFL<0.25) not satisfied (CFL={cfl_value:.3f}), reduce dt or increase dx.")

# plot
plt.plot(history['loss'])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.text(0.99*n_batch*EPOCHS, 0.08*max(history['loss']), f"ss={params[0]:.6f}\n rr={params[1]:.6f}", c='r', ha='right')
plt.grid()
plt.show()
