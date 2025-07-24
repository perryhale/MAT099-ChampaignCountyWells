import jax
import jax.numpy as jnp
import multiprocessing

# type: (np.ndarray, np.ndarray, int, bool) ~> tuple[np.ndarray, np.ndarray]
def batch_generator(data_x, data_y, batch_size, shuffle_key=None):
	
	# assertions
	assert (len(data_x)==len(data_y))
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else jnp.array(range(n_samples))
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			batch_y = data_y[batch_idx]
			yield batch_x, batch_y


# type: (np.ndarray, int, bool) ~> tuple[np.ndarray, np.ndarray]
def batch_generator_mono(data_x, batch_size, shuffle_key=None):
	
	# yield infinite batches optionally shuffled
	while True:
		n_samples = len(data_x)
		data_idx = jax.random.permutation(shuffle_key, n_samples) if (shuffle_key is not None) else jnp.array(range(n_samples))
		for batch_data_idx in range(0, n_samples, batch_size):
			batch_idx = data_idx[batch_data_idx:batch_data_idx+batch_size]
			batch_x = data_x[batch_idx]
			yield batch_x


# type: (list, (list)->list, int) -> list
def multiprocess_list(list_data, list_fn, max_threads=999):
	
	# determine thread count
	n_threads = min(multiprocessing.cpu_count(), max_threads)
	print(f"Using {n_threads} CPUs..")
	
	# prepare jobs
	job_size = len(list_data) // n_threads
	job_data = [list_data[i:i+job_size] for i in range(0, len(list_data), job_size)]
	
	# pool worker results
	with multiprocessing.Pool(n_threads) as pool:
		result = pool.map(list_fn, job_data)
		result = [item for sublist in result for item in sublist]
	
	return result
