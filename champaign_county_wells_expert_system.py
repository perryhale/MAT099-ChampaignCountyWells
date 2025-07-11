import time
import pickle
import jax
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
	h_param = w_cache['params'][0]
	h_fn = jax.vmap(lambda p,xyt: dense_neural_network(p, xyt, ha=jax.nn.relu)[0,0], in_axes=(None, 0))
	print(f"Loaded \"{W_CACHE}\"")
	print(f"[Elapsed time: {time.time()-T0:.2f}s]")

# run expert
expert_system(h_param, h_fn)
print("Closed Expert")
print(f"[Elapsed time: {time.time()-T0:.2f}s]")
