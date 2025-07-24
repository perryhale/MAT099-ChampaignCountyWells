import time
import jax
import jax.numpy as jnp
import optax
from ..data.pipeline import batch_generator

def fit_model(
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
	print(f"history_keys={list(history.keys())}")
	print(f"[Elapsed time: {time.time()-start_time:.2f}s]")
	
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
		print(f"[Elapsed time: {time.time()-start_time:.2f}s] epoch={i+1}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
	
	return (params, history)
