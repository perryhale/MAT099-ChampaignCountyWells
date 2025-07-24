import jax
import jax.numpy as jnp

def lp_norm(p, order=2):
	"""
	# type: (list[tuple[jnp.array]], int) -> float
	"""
	assert order >= 1, "\"order\" must be greater than zero."
	return jnp.sum(jnp.stack([jnp.sum(jnp.abs(leaf) ** order) for leaf in jax.tree_util.tree_leaves(p)])) ** (1.0 / order)


def loss_cce(yh, y, e=1e-9):
	"""
	# type: (jnp.array, jnp.array, float) -> float
	# yh, y inR (*, out)
	# loss inR
	"""
	return -jnp.mean(jnp.sum(y * jnp.log(yh + e), axis=-1))


def loss_mse(yh, y):
	"""
	# type: (jnp.array, jnp.array, float) -> float
	# yh, y inR (*, out)
	# loss inR
	"""
	return jnp.mean((yh-y)**2)


def accuracy_score(yh, y):
	"""	
	# yh, y inR (*, n_classes)
	# accuracy inR
	"""
	yhc = jnp.argmax(yh, axis=-1)
	yc = jnp.argmax(y, axis=-1)
	accuracy = jnp.mean(jnp.array(yhc==yc, dtype='int32'))
	return accuracy


def count_params(params):
	"""
	# type: (list[tuple[jnp.array]]) -> int
	"""
	
	if isinstance(params, jnp.ndarray):
		return jnp.prod(jnp.array(params.shape))
	elif isinstance(params, (list, tuple)):
		return jnp.sum(jnp.array([count_params(item) for item in params]))
	return 0
