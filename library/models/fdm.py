import jax
import jax.numpy as jnp
from tqdm import tqdm


@jax.jit
def darcyflow_fdm_periodic(h, k, dt, dx, dy, ss, rr):
	"""
	Solved 2D Darcy Flow PDE using explicit FDM with periodic edge boundary conditions
	
	args:
		h: jnp.array (nx, ny): Current hydraulic head
		k: jnp.array (nx, ny): Hydraulic conductivity
		dt, dx, dy: float: Discretization parameters
		ss: float: Specific-storage coefficient
		rr: float: Recharge rate
	returns:
		jnp.array (nx, ny): Next hydraulic head
	"""
	
	# solve for next state
	laplacian = (
		(jnp.roll(h, -1, axis=0) + jnp.roll(h, 1, axis=0) - 2*h) / dx**2 +
		(jnp.roll(h, -1, axis=1) + jnp.roll(h, 1, axis=1) - 2*h) / dy**2
	)
	next_h = h + dt / ss * (k * laplacian + rr)
	return next_h


@jax.jit
def darcyflow_fdm_neumann(h, k, dt, dx, dy, ss, rr):
	"""
	Solve 2D Darcy Flow PDE using explicit FDM with Neumann (zero-gradient) edge BCs
	
	args:
		h: jnp.array (nx, ny): Current hydraulic head
		k: jnp.array (nx, ny): Hydraulic conductivity
		dt, dx, dy: float: Discretization parameters
		ss: float: Specific-storage coefficient
		rr: float: Recharge rate
	returns:
		jnp.array (nx, ny): Next hydraulic head
	"""
	
	# solve for next state
	h_padded = jnp.pad(h, pad_width=1, mode='edge') # (nx+2, ny+2)
	laplacian = (
		(h_padded[2:, 1:-1] + h_padded[0:-2, 1:-1] - 2*h) / dx**2 +
		(h_padded[1:-1, 2:] + h_padded[1:-1, 0:-2] - 2*h) / dy**2
	)
	next_h = h + dt / ss * (k * laplacian + rr)
	return next_h


@jax.jit
def cfl_value(k, dt, dx, dy, ss):
	"""
	Courant–Friedrichs–Lewy simulation stability value.
	Typically, < 1/4 required for stability.
	
	args:
		k: jnp.array: 2D array representing hydraulic conductivity
		dt, dx, dy: float: Size of discretizations
		ss: float: Specific-storage coefficient
	returns:
		float: CFL value
	"""
	return jnp.max(k) * dt * (1 / dx**2 + 1 / dy**2) / ss


# type: (jnp.array, jnp.array, jnp.array) -> jnp.array
@jax.jit
def apply_dirichlet_bc(h, constraint_mask, constraint_values):
	next_h = jnp.where(constraint_mask, constraint_values, h)
	return next_h


# type: (jnp.array, jnp.array, int, float, float, float, float, float, list[(jnp.array)->jnp.array]) -> list[jnp.array]
def simulate_hydraulic_surface_fdm(h, k, n_steps, dt, dx, dy, ss, rr, dirichlet_bcs="EDGE", solver=darcyflow_fdm_neumann):
	
	# assertions
	assert h.shape == k.shape, f"ASSERT: Arrays h and k must have same shape: h.shape={h.shape}, k.shape={k.shape}."
	assert len(h.shape) == 2, f"ASSERT: Grid must be 2D: shape={h.shape}."
	
	# convenience feature
	if dirichlet_bcs=="EDGE":
		dbc_vals = jnp.zeros(h.shape, dtype='float32')
		dbc_mask = jnp.zeros(h.shape, dtype='bool')
		dbc_mask = dbc_mask.at[0, :].set(True)
		dbc_mask = dbc_mask.at[-1, :].set(True)
		dbc_mask = dbc_mask.at[:, 0].set(True)
		dbc_mask = dbc_mask.at[:, -1].set(True)
	else:
		dbc_mask, dbc_vals = dirichlet_bcs
	
	# CFL stability check
	cfl = cfl_value(k, dt, dx, dy, ss)
	if cfl >= 0.25:
		print(f"WARN: Proceeding with unstable simulation. CFL condition (CFL<0.25) not satisfied (CFL={cfl:.3f}), reduce dt or increase dx.")
	
	# iterate solver over time
	state = h
	h_sim = [h]
	for t in tqdm(range(n_steps-1)):
		state = solver(state, k, dt, dx, dy, ss, rr)
		state = apply_dirichlet_bc(state, dbc_mask, dbc_vals)
		h_sim.append(state)
	
	return h_sim
