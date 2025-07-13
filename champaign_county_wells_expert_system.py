import time
import pickle
import jax
import jax.numpy as jnp
from library.models.nn import dense_neural_network
from library.tools import expert_system


### setup

# start timer
T0 = time.time()
print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# cache path
W_CACHE = 'cache/df_rpinn.pkl'


### main

# load cache
with open(W_CACHE, 'rb') as f:
	w_cache = pickle.load(f)
	data_scaler = data_scaler = w_cache['data_scaler']
	h_param = w_cache['params'][0]
	h_fn = jax.vmap(lambda p,xyt: dense_neural_network(p, xyt, ha=jax.nn.relu)[0,0], in_axes=(None, 0))
	print(f"Loaded \"{W_CACHE}\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# run expert
expert_system(h_param, h_fn, translate=[(0,1), (0,1), (0,1), (data_scaler.data_min_[3], data_scaler.data_range_[3])])
print("Closed Expert")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
