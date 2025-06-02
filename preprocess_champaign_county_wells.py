import time
import datetime
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from library.data import (
	crop_matrix_crs,
	interpolate_hydraulic_grid
)
from library.visualize import animate_hydrology


### parameters

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

# bounding CRS
K_BOUND_N = 49.0000
K_BOUND_S = 24.0000
K_BOUND_W = -126.0000
K_BOUND_E = -66.0000 # note: longitude=x latitude=y


### functions

# type: (pd.DataFrame, pd.DataFrame) -> pd.DataFrame
def filter_well_data(data_location, data_measure):
	
	# 1. Get ['P_Number', 'TIMESTAMP', 'DTW_FT_Reviewed'] columns from data_measure & establish dataframe lendth
	data_filtered = pd.DataFrame()
	data_filtered[['P_NUMBER', 'TIMESTAMP', 'DTW_FT_Reviewed']] = data_measure[['P_Number', 'TIMESTAMP', 'DTW_FT_Reviewed']]
	
	# 2. Map over 'P_NUMBER' to get ['LONG_NAD_83', 'LAT_NAD_83', 'LS_ELEV_FT'] columns for each measurement
	data_filtered = data_filtered.merge(data_location[['P_NUMBER', 'LONG_NAD_83', 'LAT_NAD_83', 'LS_ELEV_FT']], on='P_NUMBER', how='left')
	
	# 3. Get 'HYDRAULIC_HEAD_FT' column by subtracting depth to water from land surface over sealevel
	data_filtered['HYDRAULIC_HEAD_FT'] = data_filtered['LS_ELEV_FT'] - data_filtered['DTW_FT_Reviewed']
	
	# 4. Convert date-strings to integer seconds, set start to zero
	data_filtered['TIMESTAMP'] = data_filtered['TIMESTAMP'].map(lambda usd_str: int(datetime.datetime.strptime(usd_str, "%m/%d/%Y").timestamp()))
	data_filtered['TIMESTAMP'] = data_filtered['TIMESTAMP'] - data_filtered['TIMESTAMP'].min()
	
	# 5. Average readings on the same day 'TIMESTAMP' from the same well 'P_NUMBER', cleanup types after averaging
	data_filtered['UNIQUE_ID'] = data_filtered['P_NUMBER'] + data_filtered['TIMESTAMP']
	data_filtered = data_filtered.groupby('UNIQUE_ID').mean().reset_index()
	data_filtered = data_filtered.astype({'P_NUMBER':'int32', 'TIMESTAMP':'int32'})
	
	return data_filtered


### main

# load data
k = np.array(Image.open(K_PATH))
data_location = pd.read_csv(L_PATH)
data_measure = pd.read_csv(M_PATH)
print(k.shape)
print(data_location)
print("Well#:", len(Counter(data_location['P_NUMBER'])))
print(data_measure)
print("Well#:", len(Counter(data_measure['P_Number']))) ###! only 18 wells in measurement data
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# try|create data_filtered_metric, data_surface caches
try:
	data_filtered_metric = pd.read_csv(M_CACHE)
	data_surface = pd.read_csv(S_CACHE, index_col='TIMESTAMP')
	data_surface.columns = data_surface.columns.astype('int32')
	print("Found cache")
	print(data_filtered_metric)
	print(data_surface)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

except FileNotFoundError as e:
	print("Creating cache")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# filter data
	data_filtered = filter_well_data(data_location, data_measure)
	print(data_filtered)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# convert units
	data_filtered_metric = pd.DataFrame(data_filtered[['UNIQUE_ID', 'P_NUMBER', 'TIMESTAMP', 'LONG_NAD_83', 'LAT_NAD_83']])
	data_filtered_metric['HYDRAULIC_HEAD_M'] = data_filtered['HYDRAULIC_HEAD_FT'] / 3.281 # 3.281ft ~= 1m
	print(data_filtered_metric)
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")
	
	# convert "long format" data to pivot table of well readings over time
	data_surface = data_filtered_metric.pivot(index='TIMESTAMP', columns='P_NUMBER', values='HYDRAULIC_HEAD_M').ffill()
	data_surface = data_surface.bfill() # or dropna
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

# determine well and bounding area coordinates
data_wells = data_location.set_index('P_NUMBER').loc[data_surface.columns, ['LONG_NAD_83', 'LAT_NAD_83']].to_numpy()
data_bound_n = data_wells[:,1].max()
data_bound_s = data_wells[:,1].min()
data_bound_w = data_wells[:,0].min()
data_bound_e = data_wells[:,0].max()
print("Bounding area:")
print("North:", data_bound_n)
print("South:", data_bound_s)
print("West:", data_bound_w)
print("East:", data_bound_e)
# North: 40.385156
# South: 40.05338
# West: -88.463237
# East: -87.981028

# crop k and interpolate h
k_crop, k_crop_idx = crop_matrix_crs(k, (K_BOUND_N, K_BOUND_S, K_BOUND_W, K_BOUND_E), (data_bound_n, data_bound_s, data_bound_w, data_bound_e))
grid_x, grid_y = np.meshgrid(np.linspace(data_bound_w, data_bound_e, k_crop.shape[1]), np.linspace(data_bound_n, data_bound_s, k_crop.shape[0]))
h_time = np.array([interpolate_hydraulic_grid(row, data_wells, grid_x, grid_y) for row in tqdm(data_surface.to_numpy(), desc="Grid interpolation")])
print(data_wells.shape)
print(k_crop.shape)
print(h_time.shape)
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
# (18, 2)
# (43, 48)
# (5442, 43, 48)

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
