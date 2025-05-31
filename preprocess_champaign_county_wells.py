import pandas as pd
import time
import datetime
from collections import Counter


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")


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

# type: (np.ndarray, np.ndarray) -> np.ndarray
def interpolate_grid(values, coords):
	grid_x, grid_y = np.meshgrid(
		np.linspace(coords[:,0].min()-0.0, coords[:,0].max()+0.0, 150),
		np.linspace(coords[:,1].min()-0.0, coords[:,1].max()+0.0, 50)
	)
	grid_z = interpolate.griddata(coords, values, (grid_x, grid_y), method='linear')#, fill_value=values.mean())
	
	### 1. linear interpolation
	### 2. determine NaN mask
	### 3. nearest interpolation
	### 4. combine lin interp over near interp fill NaN values
	
	return np.array([grid_x, grid_y, grid_z])


### main

# load data
data_location = pd.read_csv('data/champaign_county_wells/OB_LOCATIONS.csv')
data_measure = pd.read_csv('data/champaign_county_wells/OB_WELL_MEASUREMENTS_Champaign_County.csv')
print(data_location)
print("Well#:", len(Counter(data_location['P_NUMBER'])))
print(data_measure)
print("Well#:", len(Counter(data_measure['P_Number']))) ###! only 18 wells in measurement data
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
data_surface = data_surface.bfill() # exclusive option to dropna
#data_surface = data_surface.dropna()
print(data_surface)
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# output CSVs
data_filtered_metric.to_csv('data/champaign_county_wells/processed/data_filtered_metric.csv')
data_surface.to_csv('data/champaign_county_wells/processed/data_surface.csv')


### interpolate grids

import numpy as np
from scipy import interpolate

#fill_value = data_surface.values.mean() ###! disabled to avoid information leak
data_coords = np.array(data_location.set_index('P_NUMBER').loc[data_surface.columns][['LAT_NAD_83', 'LONG_NAD_83']].values) # note: latitude is x longtitude is y
data_x = np.array([interpolate_grid(row, data_coords) for _,row in data_surface.iterrows()])
print(data_coords.shape)
print(data_x.shape)
print(f"[Elapsed time: {time.time()-T0:.2f}s]")



print(data_filtered_metric['LONG_NAD_83'].min()) ###! coordinate mismatch
print(data_coords[1].min())
print(data_filtered_metric['LONG_NAD_83'].max())
print(data_coords[1].max())
import sys;sys.exit()

# print(data_coords[0].min())
# print(data_coords[0].max())
# print(data_coords[1].min())
# print(data_coords[1].max())
# xmin = -88.333474
# xmax = 40.116967
# ymin = -88.288459
# ymax = 40.223612
# import sys;sys.exit()


### plot interpolated grids

import matplotlib.pyplot as plt
import matplotlib.animation as animation

VIDEO_SKIP_FRAMES = 4_000
VIDEO_SAVE = False
VIDEO_INTERVAL = 10
VIDEO_FPS = 60

# load land data
k = np.load('data/SaturatedHydraulicConductivity_100m/extract_k_nad83_100m.npz')

# init plot
fig, ax = plt.subplots(figsize=(8,8))

# define frame function
# type: (int) -> None
def update(t):
	
	# skip frames
	t+=VIDEO_SKIP_FRAMES
	
	# reset axis
	ax.clear()
	
	# draw components
	im = ax.imshow(k, extent=(data_x[t][0].min(), data_x[t][0].max(), data_x[t][1].min(), data_x[t][1].max()), aspect='equal')
	contour = ax.contour(*data_x[t], levels=10, cmap='Blues')
	contour_labels = ax.clabel(contour, inline=True, fontsize=8)
	scatter = ax.scatter(data_coords[:,0], data_coords[:,1], c='red', s=12, marker='x')
	ax.set_title(f"t={t}")
	ax.set_xlabel("Longitude")
	ax.set_ylabel("Latitude")

# render animation
ani = animation.FuncAnimation(fig, update, frames=len(data_x)-VIDEO_SKIP_FRAMES, interval=VIDEO_INTERVAL)

# output
if VIDEO_SAVE:
	save_path = __file__.replace('.py','.mp4')
	ani.save(
		save_path, 
		writer='ffmpeg', 
		fps=VIDEO_FPS,
		progress_callback=lambda i,n: print(f"\33[2K\r[Frame: {i}/{n} ({100*i/n:.2f}%)][Elapsed time: {time.time()-T0:.2f}s]", end='')
	)
else:
	plt.show()


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
