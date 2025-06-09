import jax
import jax.numpy as jnp
from tqdm import tqdm


### FDM Solver

# type: (jnp.array, jnp.array, float, float, float, float, float) -> jnp.array
@jax.jit
def solve_darcy_fdm(h, k, dt, dx, dy, ss, rr):
	""" Solve 2D Darcy Flow PDE for next state using explicit finite difference method
	arg: h: jnp.array: 2D array representing current hydraulic head
	arg: k: jnp.array: 2D array representing hydraulic conductivity
	args: dt, dx, dy: float: Size of discretizations
	arg: ss: float: Specific-storage coefficient
	arg: rr: float: Recharge rate
	returns: jnp.array: Next hydraulic head
	"""
	
	# solve for next state
	laplacian = (
		(jnp.roll(h, -1, axis=0) + jnp.roll(h, 1, axis=0) - 2*h) / dx**2 +
		(jnp.roll(h, -1, axis=1) + jnp.roll(h, 1, axis=1) - 2*h) / dy**2
	)
	next_state = h + dt / ss * (k * laplacian + rr)
	
	return next_state

# type: (jnp.array, float, float, float, float) -> float
@jax.jit
def cfl_value(k, dt, dx, dy, ss):
	""" Courant–Friedrichs–Lewy simulatoin stability value, typically must be less than 1/4.
	arg: k: jnp.array: 2D array representing hydraulic conductivity
	args: dt, dx, dy: float: Size of discretizations
	arg: ss: float: Specific-storage coefficient
	returns: float: CFL value
	"""
	return jnp.max(k) * dt * (1 / dx**2 + 1 / dy**2) / ss

# type: (jnp.array) -> jnp.array
@jax.jit
def apply_edge_boundary_conditions(h):
	
	# apply edge BCs
	hhat = h
	hhat = hhat.at[0, :].set(0.0)
	hhat = hhat.at[-1, :].set(0.0)
	hhat = hhat.at[:, 0].set(0.0)
	hhat = hhat.at[:, -1].set(0.0)
	
	# apply pinhole BCs
	#hhat = hhat.at[*[n//2 for n in h.shape]].set(0.0)
	
	return hhat

# type: (jnp.array, jnp.array, int, float, float, float, float, float, List[(jnp.array)->jnp.array]) -> List[jnp.array]
def simulate_hydraulic_surface_fdm(h, k, n_steps, dt, dx, dy, ss, rr, boundary_conditions=[apply_edge_boundary_conditions]):
	
	# assertions
	assert h.shape == k.shape, f"ASSERT: Arrays h and k must have same shape: h.shape={h.shape}, k.shape={k.shape}."
	assert len(h.shape) == 2, f"ASSERT: Grid must be 2D: shape={h.shape}."
	
	# CFL stability check
	cfl = cfl_value(k, dt, dx, dy, ss)
	if cfl >= 0.25:
		print(f"WARN: Proceeding with unstable simulation. CFL condition (CFL<0.25) not satisfied (CFL={cfl:.3f}), reduce dt or increase dx.")
	
	# iterate solver over time
	state = h
	h_sim = [h]
	for t in tqdm(range(n_steps-1)):
		state = solve_darcy_fdm(state, k, dt, dx, dy, ss, rr)
		for bc_func in boundary_conditions:
			state = bc_func(state)
		h_sim.append(state)
	
	return h_sim
