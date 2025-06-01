import jax
import jax.numpy as jnp

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
