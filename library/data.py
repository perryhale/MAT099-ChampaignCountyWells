import numpy as np
import jax
import jax.numpy as jnp
from scipy import interpolate

import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping


# Perform CRS transformation and crop with rasterio/shapely
# type: (str, str, float, float, float, float) -> Tuple[np.ndarray, Tuple[int, int, int, int]]
def crop_raster(raster_path, bound_crs, bound_left, bound_low, bound_right, bound_high):
	
	# load raster data
	with rasterio.open(raster_path) as raster:
		
		# transform and crop
		bound_proj = transform_bounds(bound_crs, raster.crs, bound_left, bound_low, bound_right, bound_high)
		bound_crop,_ = mask(raster, [mapping(box(*bound_proj))], crop=True, all_touched=True)
		
		# determine crop indices
		bound_high_idx, bound_left_idx = raster.index(bound_proj[0], bound_proj[3])
		bound_low_idx, bound_right_idx = raster.index(bound_proj[2], bound_proj[1])
	
	return bound_crop[0], (bound_left_idx, bound_low_idx, bound_right_idx, bound_high_idx)

# type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
def interpolate_hydraulic_grid(values, coords, grid_x, grid_y):
	
	# 1. linear interpolation
	# 2. determine NaN mask
	# 3. nearest interpolation
	# 4. combine lin interp over near interp by filling NaN values
	grid_z_linear = interpolate.griddata(coords, values, (grid_x, grid_y), method='linear')
	#grid_z_nearest = interpolate.griddata(coords, values, (grid_x, grid_y), method='nearest')
	#grid_z_near_lin = np.where(np.isnan(grid_z_linear), grid_z_nearest, grid_z_linear) ###! causes instantaneous changes
	#grid_z_zero_lin = np.where(np.isnan(grid_z_linear), 0., grid_z_linear) ###! excesive influence
	grid_z_mean_lin = np.where(np.isnan(grid_z_linear), jnp.mean(values), grid_z_linear) ###! causes instantaneous changes
	#grid_z_rbf_mean_lin = interpolate.Rbf(grid_x, grid_y, grid_z_mean_lin, function='multiquadric') ###! unimplemented
	
	return grid_z_mean_lin

# type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, float) -> np.ndarray
def interpolate_hydraulic_grid_linear_const(values, coords, grid_x, grid_y, const):
	grid_z_linear = interpolate.griddata(coords, values, (grid_x, grid_y), method='linear')
	grid_z_linear_const = np.where(np.isnan(grid_z_linear), const, grid_z_linear)
	return grid_z_linear_const

# type: (np.ndarray, np.ndarray, int, bool) ~> Tuple[np.ndarray, np.ndarray]
def batch_generator(data_x, data_y, batch_size, shuffle_key=None):
	
	# assertions
	assert (len(data_x)==len(data_y))
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else range(n_samples)
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			batch_y = data_y[batch_idx]
			yield batch_x, batch_y


###! Deprecated

# type: (np.ndarray, Tuple[float, float, float, float], Tuple[float, float, float, float], bool) -> Tuple[np.ndarray, np.ndarray]
def crop_matrix_linear(matrix, bound, target, verbose=False):
	
	###! Does not handle curvature, results in incorrect projection
	raise(Exception("Deprecated"))
	
	# assertions
	assert len(matrix.shape) >= 2, f"Argument matrix must have at least two axis, got {len(matrix.shape)} axis."
	assert len(bound) == 4, f"Argument bound must be a 4-tuple got length {len(bound)}."
	assert len(target) == 4, f"Argument target must be a 4-tuple got length {len(target)}."
	
	# unpack
	bound_top, bound_bottom, bound_left, bound_right = bound
	target_top, target_bottom, target_left, target_right = target
	mat_width = matrix.shape[1]
	mat_height = matrix.shape[0]
	dx = (bound_right-bound_left) / mat_width
	dy = (bound_top-bound_bottom) / mat_height
	if verbose:
		print("Unpacked values:")
		print(bound)
		print(target)
		print(mat_width)
		print(mat_height)
		print(dx)
		print(dy)
	
	# project to index
	target_top_idx = (bound_top-target_top) / dy
	target_bottom_idx = (bound_top-target_bottom) / dy
	target_left_idx = (target_left-bound_left) / dx
	target_right_idx = (target_right-bound_left) / dx
	if verbose:
		print("Relative index:")
		print(target_top_idx)
		print(target_bottom_idx)
		print(target_left_idx)
		print(target_right_idx)
	target_top_idx = round(target_top_idx)
	target_bottom_idx = round(target_bottom_idx)
	target_left_idx = round(target_left_idx)
	target_right_idx = round(target_right_idx)
	if verbose:
		print("Rounded index:")
		print(target_top_idx)
		print(target_bottom_idx)
		print(target_left_idx)
		print(target_right_idx)
	
	# determine crop coordinates
	crop_x = target_left_idx
	crop_y = target_top_idx
	crop_w = target_right_idx-target_left_idx
	crop_h = target_bottom_idx-target_top_idx
	if verbose:
		print("Crop dimensions:")
		print(f"[y0,x0]: ({crop_y},{crop_x})")
		print(f"[h,w]: ({crop_h},{crop_w})")
	
	# slice matrix
	matrix_crop = matrix[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
	crop_idx = np.array((target_top_idx, target_bottom_idx, target_left_idx, target_right_idx))
	
	return matrix_crop, crop_idx


# type: (List, (List)->List, int) -> List
def multiprocess_list(list_data, list_proc_fn, max_threads=999):
	
	###! slower than list comp
	raise(Exception("Deprecated"))
	
	# determine thread count
	n_threads = min(multiprocessing.cpu_count(), max_threads)
	print(f"Using {n_threads} CPUs..")
	
	# prepare jobs
	job_size = len(list_data) // n_threads
	job_data = [list_data[i:i+job_size] for i in range(0, len(list_data), job_size)]
	
	# pool worker results
	with multiprocessing.Pool(n_threads) as pool:
		result = pool.map(list_proc_fn, job_data)
		result = [item for sublist in result for item in sublist]
	
	return result
