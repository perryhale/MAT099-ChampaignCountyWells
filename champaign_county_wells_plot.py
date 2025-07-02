import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from library.visual import animate_hydrology, plot_surface3d


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# input paths
K_PATH = 'data/SaturatedHydraulicConductivity_1km/KSat_Arithmetic_1km.tif'
S_CACHE = 'cache/data_surface.csv'
I_CACHE = 'cache/data_interpolated.npz'

# plotting arguments
PLOT_DESC = True
PLOT_DEBUG = True
PLOT_HYDRO = True
PLOT_KSAT = True
PLOT_3DSF = True

VIDEO_SAVE = False
VIDEO_FRAME_SKIP = 0


### main

# load/unpack cache
data_surface = pd.read_csv(S_CACHE)
print(f"Loaded \"{S_CACHE}\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

with np.load(I_CACHE) as data_interpolated:
	k = np.array(Image.open(K_PATH))
	k_crop = data_interpolated['k_crop']
	k_crop_idx = data_interpolated['k_crop_idx']
	grid_x = data_interpolated['grid_x']
	grid_y = data_interpolated['grid_y']
	h_time = data_interpolated['h_time']
	data_wells = data_interpolated['data_wells']
	data_bound_n = grid_y.max()
	data_bound_s = grid_y.min()
	data_bound_w = grid_x.min()
	data_bound_e = grid_x.max()
	grid_extent = (data_bound_w, data_bound_e, data_bound_s, data_bound_n)
	crop_west_idx, crop_south_idx, crop_east_idx, crop_north_idx = k_crop_idx
	print("Bounding area:")
	print("North:", data_bound_n)
	print("South:", data_bound_s)
	print("West:", data_bound_w)
	print("East:", data_bound_e)
	print(k.shape)
	print(k_crop_idx)
	print(k_crop.shape)
	print(h_time.shape)
	print(data_wells.shape)
	print(f"Loaded \"{I_CACHE}\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot descriptive statistics
if PLOT_DESC:
	data_surface.hist(bins=50, figsize=(20,15))
	plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.3, hspace=0.8)
	plt.show()

# plot debug
if PLOT_DEBUG:
	
	# grid_x increases left->right
	plt.imshow(grid_x)
	plt.title("grid_x")
	plt.show()
	print("Closed plot")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# grid_y increases lower->upper, (k increases lower->upper)
	plt.imshow(grid_y)
	plt.title("grid_y")
	plt.show()
	print("Closed plot")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# init plot
	fig, axis = plt.subplots(figsize=(10,7), nrows=2, ncols=2)
	row0, row1 = axis
	(ax01, ax02) = row0
	(ax11, ax12) = row1
	
	# plot images
	im01 = ax01.imshow(k_crop, cmap='viridis', extent=grid_extent)
	im02 = ax02.imshow(h_time[-1], cmap='Blues', extent=grid_extent)
	im11 = ax11.imshow(k_crop, cmap='viridis', aspect='equal')
	im12 = ax12.imshow(h_time[-1], cmap='Blues', aspect='equal')
	
	# scatter wells
	for ax in row0:
		ax.scatter(data_wells[:,0], data_wells[:,1], c='red', s=16, marker='x')
		for x, y in data_wells:
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
if PLOT_HYDRO:
	animate_hydrology(
		h_time,
		k=k_crop,
		grid_extent=grid_extent,
		scatter_data=data_wells.T,
		scatter_labels=True,
		xlabel="X_EPSG_6350",
		ylabel="Y_EPSG_6350",
		axis_ticks=True,
		cbar=True,
		cbar_label="cm/hr",
		frame_skip=VIDEO_FRAME_SKIP,
		save_path=__file__.replace('.py','_animation.mp4') if VIDEO_SAVE else None
	)
	print("Closed plot")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# interactive Ksat plot of well area
if PLOT_KSAT:
	plt.imshow(np.minimum(k, 999), vmin=0, vmax=25) # clipped to stop Inf error
	plt.colorbar()
	plt.scatter([crop_west_idx, crop_east_idx], [crop_north_idx, crop_south_idx], c='red', marker='+')
	plt.gca().add_patch(Rectangle(
		(crop_west_idx, crop_north_idx),
		crop_east_idx-crop_west_idx,
		crop_south_idx-crop_north_idx,
		edgecolor='red',
		facecolor=None,
		fill=False,
		lw=1
	))
	plt.tight_layout()
	plt.show()
	print("Closed plot")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# static 3d surface plot
if PLOT_3DSF:
	fig, axis = plot_surface3d(grid_x, grid_y, h_time[5140-4042], k=k_crop)
	plt.show()
