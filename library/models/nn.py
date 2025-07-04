import time
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from ..data import (
	batch_generator,
	unit_grid2_sample_fn
)


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


def get_3d_groundwater_flow_model(
		key,
		layer_dims,
		scale_xytz,
		k=jnp.ones((1,1)),
		ss=1e-1,
		rr=1e-9,
		lam_mse=1.0,
		lam_phys=1.0,
		lam_l2=0.0,
		hidden_activation=jax.nn.relu
	):
	"""
	Docstring
	"""
	
	params = [init_dense_neural_network(key, layer_dims), (ss,rr)]
	h_fn = jax.vmap(lambda p,xyt: dense_neural_network(p, xyt, ha=hidden_activation)[0,0], in_axes=(None, 0)) # N x [0,1] x [0,1] x [0,1] -> N x [0,1]
	
	def loss_3d_groundwater_flow(params, batch_xyt, scale_xytz):
		"""
		# loss = ||R||^2
		# R = Ss * ∂h/∂t - ∇·(K ∇h) - Rr
		# ∇h = (∂h/∂x, ∂h/∂y)
		# https://en.wikipedia.org/wiki/Groundwater_flow_equation
		# https://github.com/jax-ml/jax/issues/3022#issuecomment-2733591263
		"""
		
		h_fn_mono = lambda xyt: h_fn(params[0], xyt[jnp.newaxis, :])[0]
		h_fn_flux = lambda xyt: unit_grid2_sample_fn(k, *xyt[:2]) * jax.grad(h_fn_mono)(xyt)[:2] * (scale_xytz[3] / scale_xytz[:2])
		
		# compute 3d groundwater flow terms
		batch_dhdt = jax.vmap(lambda xyt: jax.grad(h_fn_mono)(xyt)[2] * (scale_xytz[3] / scale_xytz[2]))(batch_xyt)
		batch_div_flux = jax.vmap(lambda xyt: jnp.sum(jnp.diag(jax.jacfwd(h_fn_flux)(xyt)[:2, :2]) / scale_xytz[:2]))(batch_xyt)
		batch_ss = params[-1][0]
		batch_rr = params[-1][1]
		
		# compute l2 of PDE residual
		loss_darcyflow = batch_ss * batch_dhdt - batch_div_flux - batch_rr
		loss = jnp.mean(loss_darcyflow**2)
		
		return loss
	
	def loss_fn(params, batch_xyt, batch_z):
		
		loss_batch = lam_mse * loss_mse(h_fn(params[0], batch_xyt), batch_z)
		loss_phys = lam_phys * loss_3d_groundwater_flow(params, batch_xyt, scale_xytz)
		loss_reg = lam_l2 * lp_norm(params, order=2)
		loss = loss_batch + loss_phys + loss_reg
		
		return loss
	
	return (params, h_fn, loss_fn)
