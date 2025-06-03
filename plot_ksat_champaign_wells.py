import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from library.data import crop_matrix_crs


### parameters

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# input paths
K_PATH = 'data/SaturatedHydraulicConductivity_1km/KSat_Arithmetic_1km.tif'
I_CACHE = 'data/processed/data_interpolated.npz'

# bounding CRS
K_BOUND_N = 49.0000
K_BOUND_S = 24.0000
K_BOUND_W = -126.0000
K_BOUND_E = -66.0000 # note: longitude=x latitude=y

# safety switch (off)
Image.MAX_IMAGE_PIXELS = 1e16 # for 100m data


### main

# load/unpack
with np.load(I_CACHE) as data_interpolated:
	k = np.array(Image.open(K_PATH))
	k_crop_idx = data_interpolated['k_crop_idx']
	target_north_idx, target_south_idx, target_west_idx, target_east_idx = k_crop_idx
	print("Loaded cache")
	print(k.shape)
	print(k_crop_idx)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# interactive plot
plt.imshow(k, vmin=0, vmax=25)
plt.colorbar()
plt.scatter([target_west_idx, target_east_idx], [target_north_idx, target_south_idx], c='red', marker='+')
plt.gca().add_patch(Rectangle(
	(target_west_idx, target_north_idx),
	target_east_idx-target_west_idx,
	target_south_idx-target_north_idx,
	edgecolor='red',
	facecolor=None,
	fill=False,
	lw=1
))
plt.tight_layout()
plt.show()
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
