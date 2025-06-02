import numpy as np
from library.models import solve_darcy_fdm

### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# physical resolution
DX = 1 # cm
DY = DX # cm
DT = 1e-3 # hr

# cached data
I_CACHE = 'data/processed/data_interpolated.npz'
with np.load(I_CACHE) as data_interpolated:
	K_CROP = data_interpolated['k_crop']
	H_TIME = data_interpolated['h_time']
	print("Loaded cache")
	print(K_CROP.shape)
	print(H_TIME.shape)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")


### main

# partition data
h_

def loss_fn(params, x, y):
	ss, rr = params
	yh = solve_darcy_fdm(x, K_CROP, DT, DX, DY, ss, rr)
	mse_loss = np.mean((y-yh)^2)
	return mse_loss
	
