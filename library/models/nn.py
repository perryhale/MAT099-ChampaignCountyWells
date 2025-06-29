import jax
import jax.numpy as jnp
from tqdm import tqdm


### Neural Network

def init_glorot_uniform(key, shape, fan_in, fan_out):
	"""
	# type: (int, List[int], int, int) -> jnp.array
	# shape inZ+ (*)
	# fan_in, fan_out inZ+
	# w inR shape
	"""
	limit = jnp.sqrt(6) / jnp.sqrt(fan_in+fan_out)
	w = jax.random.uniform(key, shape, minval=-limit, maxval=limit)
	return w

def init_dense_neural_network(key, layers):
	"""
	# type: (int, List[int]) -> List[Tuple[jnp.array, jnp.array]]
	"""
	
	# split key
	k0, key = jax.random.split(key)
	
	# initialise input layer
	w_list = [init_glorot_uniform(k0, (layers[1], layers[0]), layers[0], layers[1])]
	b_list = [jnp.zeros((layers[1], 1))]
	
	# initialise hidden layers
	for i in range(1, len(layers)-1):
		k1, key = jax.random.split(key)
		w_list.append(init_glorot_uniform(k1, (layers[i+1], layers[i]), layers[i], layers[i+1]))
		b_list.append(jnp.zeros((layers[i+1], 1)))
	
	return list(map(tuple, zip(w_list, b_list)))

def dense_neural_network(params, x, ha=jax.nn.sigmoid):
	"""
	# type: (List[Tuple[jnp.array]], jnp.array, jax.nn.[Activation functions]) -> jnp.array
	# x inR (in, 1)
	# z inR (out, 1)
	"""
	
	# shape-safe input activation
	h = x.reshape(-1, 1)
	
	# hidden activations
	for w,b in params[:-1]:
		h = ha(jnp.dot(w, h) + b)
	
	# output activation
	wn, bn = params[-1]
	z = jnp.dot(wn, h) + bn
	
	return z

def lp_norm(p, order=2):
	"""
	# type: (List[Tuple[jnp.array]], int) -> float
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
	# type: (List[Tuple[jnp.array]]) -> int
	"""
	
	if isinstance(params, jnp.ndarray):
		return jnp.prod(jnp.array(params.shape))
	elif isinstance(params, (list, tuple)):
		return jnp.sum(jnp.array([count_params(item) for item in params]))
	return 0
