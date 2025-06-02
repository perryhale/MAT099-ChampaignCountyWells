import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from library.visualize import animate_hydrology

### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# input paths
I_CACHE = 'data/processed/data_interpolated.npz'

# plotting arguments
VIDEO_SAVE = True
VIDEO_FRAME_SKIP = 4000
DEBUG_PLOTS = True


### main

# load/unpack
with np.load(I_CACHE) as data_interpolated:
	k_crop = data_interpolated['k_crop']
	grid_x = data_interpolated['grid_x']
	grid_y = data_interpolated['grid_y']
	h_time = data_interpolated['h_time']
	data_wells = data_interpolated['data_wells']
	data_bound_n = grid_y.max()
	data_bound_s = grid_y.min()
	data_bound_w = grid_x.min()
	data_bound_e = grid_x.max()
	grid_extent = (data_bound_w, data_bound_e, data_bound_s, data_bound_n)
	print("Loaded cache")
	print("Bounding area:")
	print("North:", data_bound_n)
	print("South:", data_bound_s)
	print("West:", data_bound_w)
	print("East:", data_bound_e)
	print(data_wells.shape)
	print(k_crop.shape)
	print(h_time.shape)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

###! debug plots
if DEBUG_PLOTS:
	
	# grid_x increases left->right
	plt.imshow(grid_x)
	plt.title("grid_x")
	plt.show()
	
	# grid_y increases lower->upper, k increases lower->upper
	plt.imshow(grid_y)
	plt.title("grid_y")
	plt.show()
	
	# init plot
	fig, axis = plt.subplots(figsize=(10,7), nrows=2, ncols=2)
	row0, row1 = axis
	(ax01, ax02) = row0
	(ax11, ax12) = row1
	
	# plot images
	im01 = ax01.imshow(k_crop, aspect='equal', cmap='viridis', extent=grid_extent)
	im02 = ax02.imshow(h_time[-1], aspect='equal', cmap='Blues', extent=grid_extent)
	im11 = ax11.imshow(k_crop, aspect='equal', cmap='viridis')
	im12 = ax12.imshow(h_time[-1], aspect='equal', cmap='Blues')
	
	# scatter wells
	for ax in row0:
		ax.scatter(data_wells[:,0], data_wells[:,1], c='red', s=16, marker='x')
		for x, y in zip(data_wells[:,0], data_wells[:,1]):
			ax.text(x, y-0.01, f'({x:.2f}, {y:.2f})', color='red', fontsize=8, ha='center')
	
	# add colorbars
	fig.colorbar(im01)
	fig.colorbar(im02)
	
	# finalise
	fig.tight_layout()
	plt.show()
	print("Closed plot")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot interpolated grids
animate_hydrology(
	h_time,
	k=k_crop,
#	grid_extent=grid_extent,
#	scatter_data=data_wells.T,
	scatter_labels=True,
	xlabel="Longitude",
	ylabel="Latitude",
	axis_ticks=True,
	cbar=True,
	cbar_label="cm/hr",
	frame_skip=VIDEO_FRAME_SKIP,
	save_path=__file__.replace('.py','.mp4') if VIDEO_SAVE else None
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
