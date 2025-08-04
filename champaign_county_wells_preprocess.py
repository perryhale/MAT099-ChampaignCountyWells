import time
import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyproj

from PIL import Image
from tqdm import tqdm
from matplotlib.patches import Rectangle

from library.data.sampling import crop_raster
from library.data.interpolation import interpolate_rbf_linear
from library.visual import animate_hydrology, plot_surface3d


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# data paths
K_PATH = "data/SaturatedHydraulicConductivity_1km/KSat_Arithmetic_1km.tif"
L_PATH = "data/champaign_county_wells/OB_LOCATIONS.csv"
M_PATH = "data/champaign_county_wells/OB_WELL_MEASUREMENTS_Champaign_County.csv"

# cache paths
M_CACHE = "cache/data_filtered_metric.csv"
S_CACHE = "cache/data_surface.csv"
I_CACHE = "cache/data_interpolated"

# coordinate transform
TRANSFORM_OBJ = pyproj.Transformer.from_crs("EPSG:6319", "EPSG:6350", always_xy=True)
TRANSFORM_FN = TRANSFORM_OBJ.transform # type: (float, float) -> (float, float)

# plotting arguments
PLOT_DESC = True
PLOT_DEBUG = True
PLOT_HYDRO = True
PLOT_KSAT = True
PLOT_3DSF = True
VIDEO_SAVE = False
VIDEO_FRAME_SKIP = 0


### main

# load data
data_location = pd.read_csv(L_PATH)
print(f"Loaded \"{M_CACHE}\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
print(data_location)
print("Well#:", len(Counter(data_location['P_NUMBER'])))

data_measure = pd.read_csv(M_PATH)
print(f"Loaded \"{M_CACHE}\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
print(data_measure)
print("Well#:", len(Counter(data_measure['P_Number'])))

k = np.array(Image.open(K_PATH))
print(f"Loaded \"{K_PATH}\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
print(k.shape)

# try cache
try:
	data_filtered_metric = pd.read_csv(M_CACHE)
	print(f"Loaded \"{M_CACHE}\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	print(data_filtered_metric)
	
	data_surface = pd.read_csv(S_CACHE, index_col='TIMESTAMP')
	data_surface.columns = data_surface.columns.astype('int32')
	print(f"Loaded \"{S_CACHE}\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	print(data_surface)
	
	with np.load(I_CACHE+".npz") as data_interpolated:
		k_crop = data_interpolated['k_crop']
		k_crop_idx = data_interpolated['k_crop_idx']
		crop_west_idx, crop_south_idx, crop_east_idx, crop_north_idx = k_crop_idx
		h_rbfl = data_interpolated['h_rbfl']
		data_wells = data_interpolated['data_wells']
		grid_x = data_interpolated['grid_x']
		grid_y = data_interpolated['grid_y']
		data_bound_n = grid_y.max()
		data_bound_s = grid_y.min()
		data_bound_w = grid_x.min()
		data_bound_e = grid_x.max()
		print(f"Loaded \"{I_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
		print(f"k_crop.shape={k_crop.shape}")
		print(f"k_crop_idx={k_crop_idx}")
		print(f"h_rbfl.shape={h_rbfl.shape}")
		print(f"data_wells.shape={data_wells.shape}")
		print(f"data_bound_n={data_bound_n}")
		print(f"data_bound_s={data_bound_s}")
		print(f"data_bound_w={data_bound_w}")
		print(f"data_bound_e={data_bound_e}")
		print(f"data_wells.shape={data_wells.shape}")
		print(f"Loaded \"{I_CACHE}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	

except FileNotFoundError:
	
	# filter data
	
	# 1. Get ['P_Number', 'TIMESTAMP', 'DTW_FT_Reviewed'] columns from data_measure & establish dataframe length
	data_filtered = pd.DataFrame()
	data_filtered[['P_NUMBER', 'TIMESTAMP', 'DTW_FT_Reviewed']] = data_measure[['P_Number', 'TIMESTAMP', 'DTW_FT_Reviewed']]
	
	# 2. Map over 'P_NUMBER' to get ['LONG_NAD_83', 'LAT_NAD_83', 'LS_ELEV_FT'] columns for each measurement
	data_filtered = data_filtered.merge(data_location[['P_NUMBER', 'LONG_NAD_83','LAT_NAD_83', 'LS_ELEV_FT']], on='P_NUMBER', how='left')
	
	# 3. Get 'HYDRAULIC_HEAD_FT' column by subtracting depth to water from land surface over sealevel
	data_filtered['HYDRAULIC_HEAD_FT'] = data_filtered['LS_ELEV_FT'] - data_filtered['DTW_FT_Reviewed']
	
	# 4. Convert date-strings to integer epoch seconds
	data_filtered['TIMESTAMP'] = data_filtered['TIMESTAMP'].map(lambda us_date: int(datetime.datetime.strptime(us_date, "%m/%d/%Y").timestamp()))
	
	# 5. Average readings on the same day 'TIMESTAMP' from the same well 'P_NUMBER', cleanup types after averaging
	data_filtered['UNIQUE_ID'] = data_filtered['P_NUMBER'] + data_filtered['TIMESTAMP']
	data_filtered = data_filtered.groupby('UNIQUE_ID').mean().reset_index()
	data_filtered = data_filtered.astype({'P_NUMBER':'int32', 'TIMESTAMP':'int32'})
	print(data_filtered)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# convert units
	data_filtered_metric = pd.DataFrame(data_filtered[['UNIQUE_ID', 'P_NUMBER', 'TIMESTAMP']])
	data_filtered_metric[['X_EPSG_6350','Y_EPSG_6350']] = data_filtered[['LONG_NAD_83', 'LAT_NAD_83']].apply(lambda row : pd.Series(TRANSFORM_FN(*row)), axis=1)
	data_filtered_metric['HYDRAULIC_HEAD_M'] = data_filtered['HYDRAULIC_HEAD_FT'] / 3.281 # 3.281ft ~= 1m
	print(data_filtered_metric)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# convert "long format" data to pivot table of well readings over time
	data_surface = data_filtered_metric.pivot(index='TIMESTAMP', columns='P_NUMBER', values='HYDRAULIC_HEAD_M').ffill()
	data_surface = data_surface.bfill() # or dropna
	print(data_surface)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# truncate data_surface
	data_surface = data_surface[4042:]
	print(data_surface)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# determine well coordinates and bounding area
	data_location[['X_EPSG_6350','Y_EPSG_6350']] = data_location[['LONG_NAD_83', 'LAT_NAD_83']].apply(lambda row : pd.Series(TRANSFORM_FN(*row)), axis=1)
	data_wells = data_location.set_index('P_NUMBER').loc[data_surface.columns][['X_EPSG_6350','Y_EPSG_6350']].to_numpy()
	data_bound_n = data_wells[:,1].max()
	data_bound_s = data_wells[:,1].min()
	data_bound_w = data_wells[:,0].min()
	data_bound_e = data_wells[:,0].max()
	print(f"data_wells.shape={data_wells.shape}")
	print(f"data_bound_n={data_bound_n}")
	print(f"data_bound_s={data_bound_s}")
	print(f"data_bound_w={data_bound_w}")
	print(f"data_bound_e={data_bound_e}")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# crop and rescale k
	k_crop, k_crop_idx = crop_raster(K_PATH, "EPSG:6350", data_bound_w, data_bound_s, data_bound_e, data_bound_n)
	crop_west_idx, crop_south_idx, crop_east_idx, crop_north_idx = k_crop_idx
	#k_crop = k_crop # cm/hr
	#k_crop = k_crop * 100 # m/hr
	#k_crop = k_crop * 24e-5 # km/day
	k_crop = k_crop * 36e4**-1 # m/s
	#k_crop = np.ones(k_crop.shape) * 50 # c
	print(f"k_crop.shape={k_crop.shape}")
	print(f"k_crop_idx={k_crop_idx}")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# interpolate surface using linear RBF
	grid_x, grid_y = np.meshgrid(
		np.linspace(data_bound_w, data_bound_e, k_crop.shape[1]),
		np.linspace(data_bound_n, data_bound_s, k_crop.shape[0])
	)
	h_rbfl = np.array([interpolate_rbf_linear(row, data_wells, grid_x, grid_y) for row in tqdm(data_surface.to_numpy(), desc="Interp")])
	print(f"h_rbfl.shape={h_rbfl.shape}")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# create csv caches
	outputs = [
		(data_filtered_metric, M_CACHE),
		(data_surface, S_CACHE)
	]
	for df, path in outputs:
		df.to_csv(path)
		print(f"Saved \"{path}\"")
		print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# create npz cache
	np.savez(I_CACHE,
		data_wells=data_wells,
		k_crop=k_crop,
		k_crop_idx=k_crop_idx,
		grid_x=grid_x,
		grid_y=grid_y,
		h_rbfl=h_rbfl
	)
	print(f"Saved \"{I_CACHE}.npz\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot descriptive statistics
if PLOT_DESC:
	data_surface.hist(bins=50, figsize=(20,15))
	plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.3, hspace=0.8)
	plt.show()

# plot visual debug figures
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
	im01 = ax01.imshow(k_crop, cmap='viridis', extent=(data_bound_w, data_bound_e, data_bound_s, data_bound_n))
	im02 = ax02.imshow(h_rbfl[-1], cmap='Blues', extent=(data_bound_w, data_bound_e, data_bound_s, data_bound_n))
	im11 = ax11.imshow(k_crop, cmap='viridis', aspect='equal')
	im12 = ax12.imshow(h_rbfl[-1], cmap='Blues', aspect='equal')
	
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

# plot interpolated surface
if PLOT_HYDRO:
	animate_hydrology(
		h_rbfl,
		k=k_crop,
		grid_extent=(data_bound_w, data_bound_e, data_bound_s, data_bound_n),
		scatter_data=data_wells.T,
		scatter_labels=True,
		cmap_contour='Blues_r',
		#origin=None,
		isolines=10,
		title_fn=lambda t: f"t={t:.2f}",
		clabel_fmt='%d',
		xlabel="X_EPSG_6350",
		ylabel="Y_EPSG_6350",
		cbar=True,
		cbar_label="cm/hr",
		frame_skip=VIDEO_FRAME_SKIP,
		save_path=__file__.replace('.py','_Figure_3.mp4') if VIDEO_SAVE else None
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
	fig, axis = plot_surface3d(grid_x, grid_y, h_rbfl[5140-4042], k=k_crop)
	plt.show()
