import time

import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm

from library.models import solve_darcy_fdm


### parameters

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# grid resolution
N_STEPS = 10_000
RES_X = 64
RES_Y = RES_X

# physical resolution
DX = 1 # cm
DY = DX # cm
DT = 1e-3 # hr

# ground properties
K_PATH = 'data/SaturatedHydraulicConductivity_1km/KSat_Arithmetic_1km.tif' # hydraulic conductivity is a measure of flow distance over time. (data in centimeters per hour)
SS = 1e-1 # specific storage is a dimensionless quantity representing the volume of water released from storage per unit decline in hydraulic head
RR = 1e-3 # recharge rate is a positive contribution to the hydraulic head

# raster coordinates for upper left corner of cropped region from hydraulic conductivity data
K_CROP_X = 3250
K_CROP_Y = 1200

# plotting
VIDEO_SAVE = False
VIDEO_SKIP_FRAMES = 0#9_000


### functions

# type: (jnp.array) -> jnp.array
@jax.jit
def apply_boundary_conditions(h):
	
	# apply edge BCs
	hhat = h
	hhat = hhat.at[0, :].set(0.0)
	hhat = hhat.at[-1, :].set(0.0)
	hhat = hhat.at[:, 0].set(0.0)
	hhat = hhat.at[:, -1].set(0.0)
	
	# apply pinhole BCs
	#hhat = hhat.at[*[n//2 for n in h.shape]].set(0.0)
	
	return hhat

# type: (jnp.array, jnp.array, int, float, float, float, float, float) -> List[jnp.array]
def simulate_hydraulic_head_fdm(h, k, n_steps, dt, dx, dy, ss, rr):
	
	# assertions
	assert h.shape == k.shape, f"ASSERT: Arrays h and k must have same shape: h.shape={h.shape}, k.shape={k.shape}."
	assert len(h.shape) == 2, f"ASSERT: Grid must be 2D: shape={h.shape}."
	
	# stability check (Courant–Friedrichs–Lewy)
	cfl_value = jnp.max(k) * dt * (1 / dx**2 + 1 / dy**2) / ss
	if cfl_value >= 0.25:
		print(f"WARN: Proceeding with unstable simulation. CFL condition (CFL<0.25) not satisfied (CFL={cfl_value:.3f}), reduce dt or increase dx.\n")
	
	# iterate solver over time
	state = h
	sim_h = [h]
	for t in tqdm(range(n_steps-1)):
		state = solve_darcy_fdm(state, k, dt, dx, dy, ss, rr)
		state = apply_boundary_conditions(state)
		sim_h.append(state)
	
	return sim_h


### main

# (load/crop/clip) hydraulic conductivity data
k = jnp.array(Image.open(K_PATH))
k = k[K_CROP_Y:K_CROP_Y+RES_Y, K_CROP_X:K_CROP_X+RES_X]
k = jnp.minimum(jnp.maximum(k, 0), 10)

# run fdm simulation
init_h = jnp.ones((RES_X, RES_Y))
#init_h = jnp.array([[jnp.sin(x) for x in jnp.linspace(0,1,RES_X)] for y in jnp.linspace(0,1,RES_Y)])
sim_h = simulate_hydraulic_head_fdm(init_h, k, N_STEPS, DT, DX, DY, SS, RR)
print(f"Simulation completed.")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# animate simulation
animate_hydrology(grid_x, grid_y, sim_h, 
	k=k,
	no_ticks=True,
	skip_frames=VIDEO_SKIP_FRAMES,
	save_path=__file__.replace('.py','.mp4') if VIDEO_SAVE else None
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
