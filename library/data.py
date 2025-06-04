import math
import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate


# type: (np.ndarray, Tuple[float, float, float, float], Tuple[float, float, float, float], bool) -> Tuple[np.ndarray, np.ndarray]
def crop_matrix_linear(matrix, bound, target, verbose=False):
	
	###! Does not handle curvature, results in incorrect projection
	
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


# type: (np.ndarray, np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
def interpolate_hydraulic_grid(values, coords, grid_x, grid_y):
	
	# 1. linear interpolation
	# 2. determine NaN mask
	# 3. nearest interpolation
	# 4. combine lin interp over near interp by filling NaN values
	grid_z_linear = interpolate.griddata(coords, values, (grid_x, grid_y), method='linear')
	grid_z_nearest = interpolate.griddata(coords, values, (grid_x, grid_y), method='nearest')
	grid_z = np.where(np.isnan(grid_z_linear), grid_z_nearest, grid_z_linear)
	
	# 5. smoothing
	#grid_z_quadratic = interpolate.Rbf(grid_x, grid_y, grid_z_linear, function='multiquadric')
	#RectBivariateSpline(, kx=2, ky=2)
	
	return grid_z


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
