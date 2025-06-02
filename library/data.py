import math
import numpy as np

# type: (np.ndarray, Tuple[float, float, float, float], Tuple[float, float, float, float]) -> np.ndarray
def crop_matrix_nad83(matrix, coord_bound, coord_target, verbose=False):
	
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
	target_north_idx = math.floor(target_north_idx)
	target_south_idx = math.floor(target_south_idx)
	target_west_idx = math.floor(target_west_idx)
	target_east_idx = math.floor(target_east_idx)
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
		print(crop_x, crop_y)
		print(crop_w, crop_h)
	
	# slice matrix
	mat_cropped = matrix[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
	
	return mat_cropped, (target_north_idx, target_south_idx, target_west_idx, target_east_idx), (dx, dy)
