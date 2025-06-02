import time
import datetime
from collections import Counter

import pandas as pd
import numpy as np
from scipy import interpolate
from PIL import Image

from library.data import crop_matrix_nad83
from library.visualize import animate_hydrology


### parameters

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# hydraulic conductivity map
K_PATH = 'data/SaturatedHydraulicConductivity_1km/KSat_Arithmetic_1km.tif'
K_BOUND_N = 49.0000
K_BOUND_S = 24.0000
K_BOUND_W = -126.0000
K_BOUND_E = -66.0000 # note: longitude=x latitude=y

# plotting arguments
VIDEO_SAVE = False
VIDEO_SKIP_FRAMES = 5000


### functions

# type: (pd.DataFrame, pd.DataFrame) -> pd.DataFrame
def filter_well_data(data_location, data_measure):
	
	# initialise dataframe
	data_filtered = pd.DataFrame()
	
	# 1. Get ['P_Number', 'TIMESTAMP', 'DTW_FT_Reviewed'] columns from data_measure & establish dataframe lendth
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

# type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
def interpolate_grid(values, coords, grid_x, grid_y):
	
	# 1. linear interpolation
	# 2. determine NaN mask
	# 3. nearest interpolation
	# 4. combine lin interp over near interp by filling NaN values
	grid_z_linear = interpolate.griddata(coords, values, (grid_x, grid_y), method='linear')
	grid_z_nearest = interpolate.griddata(coords, values, (grid_x, grid_y), method='nearest')
	grid_z = np.where(np.isnan(grid_z_linear), grid_z_nearest, grid_z_linear)
	
	return grid_z


### main

# load data
k = np.array(Image.open(K_PATH))
data_location = pd.read_csv('data/champaign_county_wells/OB_LOCATIONS.csv')
data_measure = pd.read_csv('data/champaign_county_wells/OB_WELL_MEASUREMENTS_Champaign_County.csv')
print(k.shape)
print(data_location)
print("Well#:", len(Counter(data_location['P_NUMBER'])))
print(data_measure)
print("Well#:", len(Counter(data_measure['P_Number']))) ###! only 18 wells in measurement data
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# try/create cache
try:
	data_filtered_metric = pd.read_csv('data/processed/data_filtered_metric.csv')
	data_surface = pd.read_csv('data/processed/data_surface.csv', index_col='TIMESTAMP')
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
		(data_filtered_metric, "data/processed/data_filtered_metric.csv"),
		(data_surface, "data/processed/data_surface.csv")
	]
	for df, path in outputs:
		df.to_csv(path)
		print(f"Saved \"{path}\" [Elapsed time: {time.time()-T0:.2f}s]")

# determine well and bounding area coordinates
data_coords = data_location.set_index('P_NUMBER').loc[data_surface.columns, ['LONG_NAD_83', 'LAT_NAD_83']].to_numpy()
data_bound_n = data_coords[:,1].max()
data_bound_s = data_coords[:,1].min()
data_bound_w = data_coords[:,0].min()
data_bound_e = data_coords[:,0].max()
print("Bounding area:")
print("North:", data_bound_n)
print("South:", data_bound_s)
print("West:", data_bound_w)
print("East:", data_bound_e)
# North: -87.981028
# South: -88.463237
# West: 40.05338
# East: 40.385156

# crop k and interpolate h
k_cropped,_,(k_dx, k_dy) = crop_matrix_nad83(k, (K_BOUND_N, K_BOUND_S, K_BOUND_W, K_BOUND_E), (data_bound_n, data_bound_s, data_bound_w, data_bound_e), verbose=True)
grid_x, grid_y = np.meshgrid(
	np.linspace(data_bound_w, data_bound_e, k_cropped.shape[1]),
	np.linspace(data_bound_s, data_bound_n, k_cropped.shape[0])
)
h_time = np.array([interpolate_grid(row, data_coords, grid_x, grid_y) for row in data_surface.to_numpy()])
print(data_coords.shape)
print(k_cropped.shape)
print(h_time.shape)
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
# (18, 2)
# (43, 48)
# (5442, 43, 48)

# save interpolated surface data
h_time_save_path = "data/processed/champaign_county_wells_interpolated_surface"
np.savez(h_time_save_path, grid_x=grid_x, grid_y=grid_y, h_time=h_time)
print(f"Saved \"{h_time_save_path}.npz\"")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# plot interoplated grids
animate_hydrology(grid_x, grid_y, h_time, 
	k=k_cropped,
	#k_extent=(data_bound_w, data_bound_w+k_cropped.shape[1]*k_dx, data_bound_s, data_bound_s+k_cropped.shape[0]*k_dy),
	scatter_data=data_coords.T,
	xlabel="Longitude",
	ylabel="Latitude",
	cbar_label="cm/hr",
	skip_frames=VIDEO_SKIP_FRAMES,
	save_path=__file__.replace('.py','.mp4') if VIDEO_SAVE else None
)
print("Closed plot")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")


### initial time-series conversion
### creates sparse structure with surfaces for each long-format measurement

# surface = {}
# time_surface = []
# for i, row in data_filtered_metric.iterrows():
	# surface[row['P_NUMBER']] = (row['LONG_NAD_83'], row['LAT_NAD_83'], row['HYDRAULIC_HEAD_M'])
	# time_surface.append(surface)
	# print(i, len(surface.keys()))

#data_surface = pd.DataFrame([{row['P_NUMBER']:row['HYDRAULIC_HEAD_M']} for _,row in data_filtered_metric.iterrows()])
#data_surface.columns = data_surface.columns.astype(int)
#data_surface = data_surface.ffill()
#data_surface['TIMESTAMP'] = data_filtered_metric['TIMESTAMP']
