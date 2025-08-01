import time
import math
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm

from library.data import batch_generator
from library.models.fdm import (
	darcyflow_fdm_periodic,
	cfl_value,
	simulate_hydraulic_surface_fdm
)
from library.visual import animate_hydrology


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# RNG setup
RNG_SEED = 999
K0 = jax.random.key(RNG_SEED)

# cache path
I_CACHE = 'cache/data_interpolated.npz'

# solver parameters
DX = 1000
DY = DX
DT = 24
N_STEPS = 10_000

# optimizer
EPOCHS = 1
BATCH_SIZE = 16
ETA = 1e-6
RHO = 25e-2

# plotting
VIDEO_FRAME_SKIP = 0
VIDEO_SAVE = False


### main

# load cache
with jnp.load(I_CACHE) as data_interpolated:
	k_crop = data_interpolated['k_crop']
	h_time = data_interpolated['h_time']
	print("Loaded cache")
	print(k_crop.shape)
	print(h_time.shape)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# time-series supervised alignment
data_x = h_time[:-1]
data_y = h_time[1:]
print(data_x.shape)
print(data_y.shape)

# define model
params = jnp.array((0.1, 0.1))
model = jax.vmap(lambda p,x: darcyflow_fdm_periodic(x, k_crop, DT, DX, DY, p[0], p[1]), in_axes=(None, 0))
loss_fn = lambda p,x,y: jnp.mean(jnp.pow(y - model(p, x), 2))
print(params)

# setup optimizer
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
		cfl = cfl_value(k_crop, DT, DX, DY, params[0])
		if cfl >= 0.25:
			print(f"WARN: Proceeding with unstable simulation. CFL condition (CFL<0.25) not satisfied (CFL={cfl:.3f}), reduce dt or increase dx.")
		
		# compute loss and gradient
		batch_x, batch_y = next(data_generator)
		batch_loss, state, params = optimizer_step(state, params, batch_x, batch_y)
		
		# record
		history['loss'].append(batch_loss)
		history['params'].append(params)
		print(f"[Elapsed time: {time.time()-T0:.2f}s] epoch={i+1}, batch={j+1}, loss={batch_loss}, params={params}")

# plot optimizer history
plt.plot(history['loss'])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.text(0.99*n_batch*EPOCHS, 0.08*max(history['loss']), f"ss={params[0]:.6f}\n rr={params[1]:.6f}", c='r', ha='right')
plt.grid()
plt.show()

# simulate
#h_init = data_y[-1]
h_init = jnp.ones(k_crop.shape)
#h_init = jnp.array([[jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y) for x in jnp.linspace(0, 1, k_crop.shape[1])] for y in jnp.linspace(0, 1, k_crop.shape[0])]) # central blob
h_sim = simulate_hydraulic_surface_fdm(h_init, k_crop, N_STEPS, DT, DX, DY, *params)
print(f"Simulation completed.")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

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
