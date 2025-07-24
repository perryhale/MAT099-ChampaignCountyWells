import jax
import jax.numpy as jnp
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds
from shapely.geometry import box, mapping

# type: (jnp.array, float, float) -> any
def unit_grid2_sample_fn(grid2, x, y):
	x = jnp.clip(x, 0, 1)
	y = jnp.clip(y, 0, 1)
	x_idx = jnp.floor(x * grid2.shape[1]-1).astype(jnp.int32)
	y_idx = jnp.floor(y * grid2.shape[0]-1).astype(jnp.int32)
	return jax.lax.dynamic_slice(grid2, (y_idx, x_idx), (1, 1))[0, 0] # https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html


# type: (str, str, float, float, float, float) -> tuple[np.ndarray, tuple[int, int, int, int]]
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


###! 

# # type: (np.ndarray, tuple[float, float, float, float], tuple[float, float, float, float], bool) -> tuple[np.ndarray, np.ndarray]
# def crop_matrix_linear(matrix, bound, target, verbose=False):
	
	# # assertions
	# assert len(matrix.shape) >= 2, f"Argument matrix must have at least two axis, got {len(matrix.shape)} axis."
	# assert len(bound) == 4, f"Argument bound must be a 4-tuple got length {len(bound)}."
	# assert len(target) == 4, f"Argument target must be a 4-tuple got length {len(target)}."
	
	# # unpack
	# bound_top, bound_bottom, bound_left, bound_right = bound
	# target_top, target_bottom, target_left, target_right = target
	# mat_width = matrix.shape[1]
	# mat_height = matrix.shape[0]
	# dx = (bound_right-bound_left) / mat_width
	# dy = (bound_top-bound_bottom) / mat_height
	# if verbose:
		# print("Unpacked values:")
		# print(bound)
		# print(target)
		# print(mat_width)
		# print(mat_height)
		# print(dx)
		# print(dy)
	
	# # project to index
	# target_top_idx = (bound_top-target_top) / dy
	# target_bottom_idx = (bound_top-target_bottom) / dy
	# target_left_idx = (target_left-bound_left) / dx
	# target_right_idx = (target_right-bound_left) / dx
	# if verbose:
		# print("Relative index:")
		# print(target_top_idx)
		# print(target_bottom_idx)
		# print(target_left_idx)
		# print(target_right_idx)
	# target_top_idx = round(target_top_idx)
	# target_bottom_idx = round(target_bottom_idx)
	# target_left_idx = round(target_left_idx)
	# target_right_idx = round(target_right_idx)
	# if verbose:
		# print("Rounded index:")
		# print(target_top_idx)
		# print(target_bottom_idx)
		# print(target_left_idx)
		# print(target_right_idx)
	
	# # determine crop coordinates
	# crop_x = target_left_idx
	# crop_y = target_top_idx
	# crop_w = target_right_idx-target_left_idx
	# crop_h = target_bottom_idx-target_top_idx
	# if verbose:
		# print("Crop dimensions:")
		# print(f"[y0,x0]: ({crop_y},{crop_x})")
		# print(f"[h,w]: ({crop_h},{crop_w})")
	
	# # slice matrix
	# matrix_crop = matrix[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
	# crop_idx = jnp.array((target_top_idx, target_bottom_idx, target_left_idx, target_right_idx))
	
	# return matrix_crop, crop_idx
