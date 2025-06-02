import math
import numpy as np
from scipy import interpolate


# type: (np.ndarray, Tuple[float, float, float, float], Tuple[float, float, float, float], bool) -> Tuple[np.ndarray, np.ndarray]
def crop_matrix_crs(matrix, coord_bound, coord_target, verbose=False):
	
	###! Does not handle curvature, results in incorrect projection
	
	# assertions
	assert len(matrix.shape) >= 2, f"Argument matrix must have at least two axis, got {len(matrix.shape)} axis."
	assert len(coord_bound) == 4, f"Argument coord_bound must be a 4-tuple got length {len(coord_bound)}."
	assert len(coord_target) == 4, f"Argument coord_target must be a 4-tuple got length {len(coord_target)}."
	
	# unpack
	bound_north, bound_south, bound_west, bound_east = coord_bound
	target_north, target_south, target_west, target_east = coord_target
	mat_width = matrix.shape[1]
	mat_height = matrix.shape[0]
	dx = (bound_east-bound_west) / mat_width
	dy = (bound_north-bound_south) / mat_height
	if verbose:
		print("crop_matrix_nad83: Unpack")
		print(coord_bound)
		print(coord_target)
		print(mat_width)
		print(mat_height)
		print(dx)
		print(dy)
	
	# project to index
	target_north_idx = (bound_north-target_north) / dy
	target_south_idx = (bound_north-target_south) / dy
	target_west_idx = (target_west-bound_west) / dx
	target_east_idx = (target_east-bound_west) / dx
	if verbose:
		print("crop_matrix_nad83: Index projection")
		print(target_north_idx)
		print(target_south_idx)
		print(target_west_idx)
		print(target_east_idx)
	target_north_idx = round(target_north_idx)
	target_south_idx = round(target_south_idx)
	target_west_idx = round(target_west_idx)
	target_east_idx = round(target_east_idx)
	if verbose:
		print("crop_matrix_nad83: Index rounding")
		print(target_north_idx)
		print(target_south_idx)
		print(target_west_idx)
		print(target_east_idx)
	
	# determine crop coordinates
	crop_x = target_west_idx
	crop_y = target_north_idx
	crop_w = target_east_idx-target_west_idx
	crop_h = target_south_idx-target_north_idx
	if verbose:
		print("crop_matrix_nad83: Crop coordinates")
		print(f"({crop_y},{crop_x})")
		print(f"({crop_h},{crop_w})")
	
	# slice matrix
	matrix_crop = matrix[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
	crop_idx = np.array((target_north_idx, target_south_idx, target_west_idx, target_east_idx))
	
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
