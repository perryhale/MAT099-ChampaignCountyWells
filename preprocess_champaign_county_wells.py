import time
import datetime
from collections import Counter

import numpy as np
import pandas as pd
import pyproj
from PIL import Image
from tqdm import tqdm

from library.data import *
from library.models.fdm import *


### Parameters

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# input paths
K_PATH = 'data/SaturatedHydraulicConductivity_1km/KSat_Arithmetic_1km.tif'
L_PATH = 'data/champaign_county_wells/OB_LOCATIONS.csv'
M_PATH = 'data/champaign_county_wells/OB_WELL_MEASUREMENTS_Champaign_County.csv'

# output paths
M_CACHE = "data/processed/data_filtered_metric.csv"
S_CACHE = "data/processed/data_surface.csv"
I_CACHE = "data/processed/data_interpolated"


### Main section

# load data
data_location = pd.read_csv(L_PATH)
data_measure = pd.read_csv(M_PATH)
print(data_location)
print("Well#:", len(Counter(data_location['P_NUMBER'])))
print(data_measure)
print("Well#:", len(Counter(data_measure['P_Number']))) ###! only 18 wells in measurement data
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

###! transform coordinates from NAD83 (EPSG:6319) to Conus Albers (EPSG:6350)
transform_obj = pyproj.Transformer.from_crs("EPSG:6319", "EPSG:6350", always_xy=True)
transform_fn = transform_obj.transform # type: (float, float) -> (float, float)

# try cache
try:
	data_filtered_metric = pd.read_csv(M_CACHE)
	data_surface = pd.read_csv(S_CACHE, index_col='TIMESTAMP')
	data_surface.columns = data_surface.columns.astype('int32')
	print("Found cache")
	print(data_filtered_metric)
	print(data_surface)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# create data_filtered_metric, data_surface caches
except FileNotFoundError as e:
	print("Creating cache")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# filter data
	
	# 1. Get ['P_Number', 'TIMESTAMP', 'DTW_FT_Reviewed'] columns from data_measure & establish dataframe length
	data_filtered = pd.DataFrame()
	data_filtered[['P_NUMBER', 'TIMESTAMP', 'DTW_FT_Reviewed']] = data_measure[['P_Number', 'TIMESTAMP', 'DTW_FT_Reviewed']]
	
	# 2. Map over 'P_NUMBER' to get ['LONG_NAD_83', 'LAT_NAD_83', 'LS_ELEV_FT'] columns for each measurement
	data_filtered = data_filtered.merge(data_location[['P_NUMBER', 'LONG_NAD_83','LAT_NAD_83', 'LS_ELEV_FT']], on='P_NUMBER', how='left')
	
	# 3. Get 'HYDRAULIC_HEAD_FT' column by subtracting depth to water from land surface over sealevel
	data_filtered['HYDRAULIC_HEAD_FT'] = data_filtered['LS_ELEV_FT'] - data_filtered['DTW_FT_Reviewed']
	
	# 4. Convert date-strings to integer epoch seconds, set start to zero
	data_filtered['TIMESTAMP'] = data_filtered['TIMESTAMP'].map(lambda usd_str: int(datetime.datetime.strptime(usd_str, "%m/%d/%Y").timestamp()))
	data_filtered['TIMESTAMP'] = data_filtered['TIMESTAMP'] - data_filtered['TIMESTAMP'].min()
	
	# 5. Average readings on the same day 'TIMESTAMP' from the same well 'P_NUMBER', cleanup types after averaging
	data_filtered['UNIQUE_ID'] = data_filtered['P_NUMBER'] + data_filtered['TIMESTAMP']
	data_filtered = data_filtered.groupby('UNIQUE_ID').mean().reset_index()
	data_filtered = data_filtered.astype({'P_NUMBER':'int32', 'TIMESTAMP':'int32'})
	print(data_filtered)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# convert units
	data_filtered_metric = pd.DataFrame(data_filtered[['UNIQUE_ID', 'P_NUMBER', 'TIMESTAMP', 'LONG_NAD_83','LAT_NAD_83']])
	data_filtered_metric[['X_EPSG_6350','Y_EPSG_6350']] = data_filtered_metric[['LONG_NAD_83', 'LAT_NAD_83']].apply(lambda row : pd.Series(transform_fn(*row)), axis=1)
	data_filtered_metric['HYDRAULIC_HEAD_M'] = data_filtered['HYDRAULIC_HEAD_FT'] / 3.281 # 3.281ft ~= 1m
	print(data_filtered_metric)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# convert "long format" data to pivot table of well readings over time
	data_surface = data_filtered_metric.pivot(index='TIMESTAMP', columns='P_NUMBER', values='HYDRAULIC_HEAD_M').ffill()
	data_surface = data_surface.bfill() # or dropna
	print(data_surface)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	###! truncate data_surface
	data_surface = data_surface[4042:]
	print(data_surface)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# output CSVs
	outputs = [
		(data_filtered_metric, M_CACHE),
		(data_surface, S_CACHE)
	]
	for df, path in outputs:
		df.to_csv(path)
		print(f"Saved \"{path}\" [Elapsed time: {time.time()-T0:.2f}s]")

# determine well coordinates and bounding area
data_location[['X_EPSG_6350','Y_EPSG_6350']] = data_location[['LONG_NAD_83', 'LAT_NAD_83']].apply(lambda row : pd.Series(transform_fn(*row)), axis=1)
data_wells = data_location.set_index('P_NUMBER').loc[data_surface.columns][['X_EPSG_6350','Y_EPSG_6350']].to_numpy()
data_bound_n = data_wells[:,1].max()
data_bound_s = data_wells[:,1].min()
data_bound_w = data_wells[:,0].min()
data_bound_e = data_wells[:,0].max()
print("Bounding area:")
print("North:", data_bound_n)
print("South:", data_bound_s)
print("West:", data_bound_w)
print("East:", data_bound_e)

# crop and rescale k
k_crop, k_crop_idx = crop_raster(K_PATH, "EPSG:6350", data_bound_w, data_bound_s, data_bound_e, data_bound_n)
#k_crop = k_crop # cm/hr
#k_crop = k_crop * 24e-5 # km/day
#k_crop = k_crop * 36e4**-1 # m/s
k_crop = k_crop * 100 # m/hr
#k_crop = np.ones(k_crop.shape) * 50 ###! constant K
print(k_crop.mean())
print(k_crop.var())

# interpolate h time
grid_x, grid_y = np.meshgrid(
	np.linspace(data_bound_w, data_bound_e, k_crop.shape[1]),
	np.linspace(data_bound_n, data_bound_s, k_crop.shape[0])
)

###! interpolate surface using linear simplex with constant boundary condition
#global_mean = data_surface.mean().mean() ###! using global mean down-task will cause information leak
#h_time = np.array([interp2d_ls(row, data_wells, grid_x, grid_y, const=global_mean) for row in tqdm(data_surface.to_numpy(), desc="Interp")])

###! interpolate surface using linear RBF
h_time = np.array([interp2d_lrbf(row, data_wells, grid_x, grid_y) for row in tqdm(data_surface.to_numpy(), desc="Interp")])

###! interpolate surface using Dirichlet constrained FDM equilibrium
# import matplotlib.pyplot as plt
# from library.visualize import plot_surface3d
# solver = darcyflow_fdm_neumann
# grid_extent = (data_bound_w, data_bound_e, data_bound_n, data_bound_s)
# max_iter = 10_000
# dx = 1000
# dy = dx
# dt = 24
# ss = 1e-1#1.46e-1
# rr = 1e-7#7.41e-5
# def interpolate_fdm_constrained(solver, values, coords, grid_extent, grid_shape, max_iter, k, dt, dx, dy, ss, rr):
	# min_x, max_x, min_y, max_y = grid_extent
	
	# # define boundary condition
	# dbc_mask = np.zeros(grid_shape, dtype='bool')
	# dbc_vals = np.zeros(grid_shape, dtype='float')
	# for (x,y),z in zip(coords, values):
		# y_idx = int((y - min_y) / (max_y - min_y) * (dbc_mask.shape[0]-1))
		# x_idx = int((x - min_x) / (max_x - min_x) * (dbc_mask.shape[1]-1))
		# dbc_mask[y_idx, x_idx] = True
		# dbc_vals[y_idx, x_idx] = z
	
	# # iterate solver
	# grid_z = np.ones(grid_x.shape) * np.mean(values)#np.zeros(grid_x.shape)
	# for i in range(max_iter):
	# #for i in tqdm(range(max_iter)): ###! debug (tqdm)
		# grid_z = solver(grid_z, k, dt, dx, dy, ss, rr)
		# grid_z = apply_dirichlet_bc(grid_z, dbc_mask, dbc_vals)
		# ###! debug
		# # if (i % 250 == 0):
			# # _ = plot_surface3d(grid_x, grid_y, grid_z, k=k)
			# # plt.tight_layout()
			# # plt.savefig(f"figures/{time.time()}.png")
			# # plt.close()
			# # plt.imshow(grid_z)
			# # plt.colorbar()
			# # plt.savefig(f"figures/{time.time()}.png")
			# # plt.close()
	
	# ###! debug
	# # _ = plot_surface3d(grid_x, grid_y, grid_z, k=k)
	# # plt.tight_layout()
	# # plt.savefig(f"figures/{time.time()}.png")
	# # plt.close()
	# # plt.imshow(grid_z)
	# # plt.colorbar()
	# # plt.savefig(f"figures/{time.time()}.png")
	# # plt.close()
	
	# return grid_z

# h_time = np.array([
	# interpolate_fdm_constrained(solver, row, data_wells, grid_extent, grid_x.shape, max_iter, k_crop, dt, dx, dy, ss, rr) for row in tqdm(data_surface.to_numpy(), desc="Interp")
# ])

print(data_wells.shape)
print(k_crop.shape)
print(h_time.shape)
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# create cache
np.savez(I_CACHE,
	data_wells=data_wells,
	k_crop=k_crop,
	k_crop_idx=k_crop_idx,
	grid_x=grid_x,
	grid_y=grid_y,
	h_time=h_time
)
print(f"Saved \"{I_CACHE}.npz\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
