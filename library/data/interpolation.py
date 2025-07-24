import jax.numpy as jnp
import scipy.interpolate as interpolate
from ..models.fdm import apply_dirichlet_bc

# type: (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, float) -> jnp.ndarray
def interpolate_linear(values, coords, grid_x, grid_y, const=None):
	
	grid_z = interpolate.griddata(coords, values, (grid_x, grid_y), method='linear')
	grid_z = jnp.where(jnp.isnan(grid_z), jnp.mean(values) if const is None else const, grid_z)
	
	return grid_z


# type: (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray) -> jnp.ndarray
def interpolate_rbf_linear(values, coords, grid_x, grid_y):
	
	grid_flat = jnp.column_stack([grid_x.ravel(), grid_y.ravel()]) # (n, 2)
	lrbf = interpolate.RBFInterpolator(coords, values, kernel='linear')
	grid_z = lrbf(grid_flat).reshape(grid_x.shape)
	
	return grid_z


# type: ((jnp.array)->jnp.array, list[float*3], jnp.array, jnp.array, int, (int, jnp.array*3)->None)
def interpolate_fdm_constrained(solver, coords_xyz, grid_x, grid_y, max_iter, callback=None):
	
	assert grid_x.shape==grid_y.shape, f"grid_x.shape!=grid_y.shape, {grid_x.shape}!={grid_y.shape}"
	min_x, max_x = grid_x.min(), grid_x.max()
	min_y, max_y = grid_y.min(), grid_y.max()
	
	# define boundary condition
	dbc_mask = jnp.zeros(grid_x.shape, dtype='bool')
	dbc_vals = jnp.zeros(grid_x.shape, dtype='float')
	for x,y,z in coords_xyz:
		y_idx = int((y - min_y) / (max_y - min_y) * (dbc_mask.shape[0]-1))
		x_idx = int((x - min_x) / (max_x - min_x) * (dbc_mask.shape[1]-1))
		dbc_mask[y_idx, x_idx] = True
		dbc_vals[y_idx, x_idx] = z
	
	# iterate solver
	grid_z = jnp.ones(grid_x.shape) * jnp.mean(values)
	for i in range(max_iter):
		if callback is not None: callback(i, grid_x, grid_y, grid_z)
		grid_z = apply_dirichlet_bc(grid_z, dbc_mask, dbc_vals)
		grid_z = solver(grid_z)
	
	if callback is not None: callback(i, grid_x, grid_y, grid_z)
	
	return grid_z


###!

# from sklearn import gaussian_process
# # type: (jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray) -> jnp.ndarray
# def interp2d_gpr_rbf(values, coords, grid_x, grid_y):
	
	# # values_mean = values.mean()
	# # values_std = values.std()
	# # values = (values - values_mean) / values_std
	
	# #kernel = 1 * gaussian_process.kernels.RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e2))
	# gpr = gaussian_process.GaussiajnprocessRegressor()
	# gpr.fit(coords, values)
	
	# grid_flat = jnp.column_stack([grid_x.ravel(), grid_y.ravel()])
	# grid_z = gpr.predict(grid_flat).reshape(grid_x.shape)
	
	# # grid_z = grid_z * values_std + values_mean
	
	# return grid_z

# from pykrige.ok import OrdinaryKriging
# def interp2d_ok(values, coords, grid_x, grid_y):
	
	# ordinary_kriging = OrdinaryKriging(
		# coords[:,0],
		# coords[:,1],
		# values,
		# variogram_model='linear',
		# verbose=True
	# )
	# grid_z, ss = ordinary_kriging.execute('grid', jnp.linspace(grid_x.min(), grid_x.max(), grid_x.shape[1]), jnp.linspace(grid_y.max(), grid_y.min(), grid_y.shape[0]))
	
	# import matplotlib.pyplot as plt
	
	# fig = plt.figure(figsize=(5, 5))
	# ax = fig.add_subplot(111, projection='3d')
	# ax.plot_surface(grid_x, grid_y, grid_z, cmap='prism', alpha=0.8, linewidth=0)
	# contours = ax.contour(
		# grid_x, grid_y, grid_z,
		# zdir='z',
		# offset=jnp.min(grid_z)-1,
		# levels=10,
		# cmap='Oranges'
	# )
	# ax.clabel(contours, fmt='%1.1f', colors='black', fontsize=8)
	# ax.view_init(elev=20, azim=270)
	# ax.set_xlabel('X')
	# ax.set_ylabel('Y')
	# ax.set_zlabel('Z')
	# plt.tight_layout()
	# plt.show()
	
	# return grid_z
