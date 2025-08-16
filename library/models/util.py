import time
import jax
import jax.numpy as jnp
import optax
from ..data.pipeline import batch_generator

def fit(
		key,
		params,
		loss_fn,
		train_data,
		val_data=None,
		batch_size=64,
		epochs=1,
		opt=optax.sgd(1e-3),
		start_time=None
	):
	"""
	Docstring
	"""
	
	# unpack/initialise
	train_x, train_y, train_steps = train_data
	opt_state = opt.init(params)
	history = {'batch_loss':[], 'train_loss':[]}
	if val_data is not None:
		val_x, val_y, val_steps = val_data
		val_generator = batch_generator(val_x, val_y, batch_size)
		history.update({'val_loss':[]})
	if start_time is None:
		start_time = time.time()
	
	@jax.jit
	def opt_step(opt_state_, params_, x, y):
		loss, grad = jax.value_and_grad(loss_fn)(params_, x, y)
		updates, opt_state_ = opt.update(grad, opt_state_, params_)
		params_ = optax.apply_updates(params_, updates)
		return loss, opt_state_, params_
	
	# fit model
	for i in range(epochs):
		
		# shuffle train batches
		subkey, key = jax.random.split(key, 2)
		train_generator = batch_generator(train_x, train_y, batch_size, shuffle_key=subkey)
		
		# iterate optimiser
		train_loss = 0.
		for j in range(train_steps):
			batch_loss, opt_state, params = opt_step(opt_state, params, *next(train_generator))
			train_loss += batch_loss / train_steps
			
			history['batch_loss'].append(batch_loss)
		
		history['train_loss'].append(train_loss)
		
		# determine validation loss
		if val_data is not None:
			val_loss = 0.
			for _ in range(val_steps):
				val_loss += loss_fn(params, *next(val_generator)) / val_steps
			
			history['val_loss'].append(val_loss)
		
		# trace
		print(f"epoch={i+1}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f} [Elapsed time: {time.time()-start_time:.2f}s]")
	
	return (params, history)


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
