import numpy as np

# type: (np.ndarray, Tuple[float, float, float, float], Tuple[float, float, float, float]) -> np.ndarray
def crop_matrix_nad83(matrix, coord_bound, coord_target):
	
	# unpack
	bound_north, bound_south, bound_west, bound_east = coord_bound
	target_north, target_south, target_west, target_east = coord_target
	mat_width = matrix.shape[1]
	mat_height = matrix.shape[0]
	dx = (bound_east-bound_west) / mat_width
	dy = (bound_north-bound_south) / mat_height
	
	# project to index
	target_north_idx = int((bound_north-target_north) / dy)
	target_south_idx = int((bound_north-target_south) / dy)
	target_west_idx = int((target_west-bound_west) / dx)
	target_east_idx = int((target_east-bound_west) / dx)
	
	# determine crop coordinates
	crop_x = target_west_idx
	crop_y = target_north_idx
	crop_w = target_east_idx-target_west_idx
	crop_h = target_south_idx-target_north_idx
	
	# slice matrix
	mat_cropped = matrix[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
	
	return mat_cropped, (target_north_idx, target_south_idx, target_west_idx, target_east_idx)
