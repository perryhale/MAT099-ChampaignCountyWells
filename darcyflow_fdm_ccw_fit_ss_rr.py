import time
import math
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from library.data import batch_generator
from library.models import solve_darcy_fdm, cfl_value


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
BATCH_SIZE = 32
ETA = 1e-6
RHO = 25e-2


### main

# load cache
with jnp.load(I_CACHE) as data_interpolated:
	k_crop = data_interpolated['k_crop']
	h_time = data_interpolated['h_time']
	print("Loaded cache")
	print(k_crop.shape)
	print(h_time.shape)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# # rescale K
# k_crop = k_crop * 24e-5 # km/day
# print(k_scaled.mean())
# print(k_scaled.var())

# truncate+partition data
data_x = h_time[4_041:][:-1]
data_y = h_time[4_041:][1:]
print(data_x.shape)
print(data_y.shape)

# define model
params = jnp.array((0.1, 0.1))
model = jax.vmap(lambda p,x: solve_darcy_fdm(x, k_crop, DT, DX, DY, p[0], p[1]), in_axes=(None, 0))
loss_fn = lambda p,x,y: jnp.mean(jnp.pow(y - model(p, x), 2))
print(params)

# setup optimzer
optim = optax.sgd(ETA, momentum=RHO)
state = optim.init(params)
epoch_key = K0
n_batch = math.ceil(data_x.shape[0] / BATCH_SIZE)
history = {'loss':[], 'params':[]}
print(n_batch)

@jax.jit
def optimizer_step(s, p, x, y):
	loss, grad = jax.value_and_grad(loss_fn)(p, x, y)
	updates, next_state = optim.update(grad, s, p)
	next_params = optax.apply_updates(p, updates)
	return loss, next_state, next_params

# fit model
for i in range(EPOCHS):
	
	# setup data generator
	epoch_key = jax.random.split(epoch_key, 1)[0]
	data_generator = batch_generator(data_x, data_y, BATCH_SIZE, shuffle_key=epoch_key)
	
	# iterate optimizer
	for j in range(n_batch):
		
		# cfl stability check
		if cfl_value(k_crop, DT, DX, DY, params[0]) >= 0.25:
			print(f"WARN: Proceeding with unstable simulation. CFL condition (CFL<0.25) not satisfied (CFL={cfl_value:.3f}), reduce dt or increase dx.")
		
		# compute loss and gradient
		batch_x, batch_y = next(data_generator)
		batch_loss, state, params = optimizer_step(state, params, batch_x, batch_y)
		
		# record
		history['loss'].append(batch_loss)
		history['params'].append(params)
		print(f"[Elapsed time: {time.time()-T0:.2f}s] epoch={i+1}, batch={j+1}, loss={batch_loss}, params={params}")

# plot
plt.plot(history['loss'])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.text(0.99*n_batch*EPOCHS, 0.08*max(history['loss']), f"ss={params[0]:.6f}\n rr={params[1]:.6f}", c='r', ha='right')
plt.grid()
plt.show()
