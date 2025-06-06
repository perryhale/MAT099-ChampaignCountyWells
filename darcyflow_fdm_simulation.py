import time
import jax
import jax.numpy as jnp
from PIL import Image
from tqdm import tqdm

from library.models import (
	solve_darcy_fdm,
	cfl_value,
	simulate_hydraulic_surface_fdm
)
from library.visualize import animate_hydrology


### parameters

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
I_CACHE = 'data/processed/data_interpolated.npz'

# grid scale
DX = 1000 # m
DY = DX # m
DT = 3600 # s

# simulation constants
N_STEPS = 1000
SS = 0.259
RR = 1e-6

# plotting
VIDEO_SAVE = False
VIDEO_FRAME_SKIP = 0#5_000

# ground properties
#K_PATH = 'data/SaturatedHydraulicConductivity_1km/KSat_Arithmetic_1km.tif' # hydraulic conductivity is a measure of flow distance over time. (data in centimeters per hour)
#SS = 0.1 # specific storage is a quantity representing the volume of water released from storage per unit decline in hydraulic head. SI unit: inverse length
#RR = 1e-3 # recharge rate is a positive contribution to the hydraulic head


### main

# load cache
with jnp.load(I_CACHE) as data_interpolated:
	k_crop = data_interpolated['k_crop']
	print("Loaded cache")
	print(k_crop.shape)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# rescale K
#k_crop = k_crop # cm/hr
#k_crop = k_crop * 24e-5 # km/day
k_crop = k_crop * 36e4**-1 # m/s
print(k_crop.mean())
print(k_crop.var())

# run fdm simulation
#init_h = jnp.ones(k_crop.shape)
init_h = jnp.array([[jnp.sin(jnp.pi*x)*jnp.sin(jnp.pi*y) for x in jnp.linspace(0, 1, k_crop.shape[1])] for y in jnp.linspace(0, 1, k_crop.shape[0])])
h_sim = simulate_hydraulic_surface_fdm(init_h, k_crop, N_STEPS, DT, DX, DY, SS, RR)
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
