import jax
import jax.numpy as jnp
from sklearn.preprocessing import MinMaxScaler

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


# # type: (list, (list)->list, int) -> list
# import multiprocessing
# def multiprocess_list(list_data, list_fn, max_threads=999):
	
	# # determine thread count
	# n_threads = min(multiprocessing.cpu_count(), max_threads)
	# print(f"Using {n_threads} CPUs..")
	
	# # prepare jobs
	# job_size = len(list_data) // n_threads
	# job_data = [list_data[i:i+job_size] for i in range(0, len(list_data), job_size)]
	
	# # pool worker results
	# with multiprocessing.Pool(n_threads) as pool:
		# result = pool.map(list_fn, job_data)
		# result = [item for sublist in result for item in sublist]
	
	# return result

# type: (List[Tuple[float*4]], float, float, float, float) -> Tuple(Tuple(jnp.array*2, int)*3, MinMaxScaler)
def train_val_test_split(key, data_points, batch_size, part_buffer=0.0, part_train=0.7, part_val=0.05, part_test=0.25):
	
	data_points = jnp.array(data_points)
	
	# time-series partition with buffer, shuffle val+test together
	n_data = data_points.shape[0]
	n_buffer = int(part_buffer * n_data)
	n_train = int(part_train * n_data)
	n_val = int(part_val * n_data)
	n_test = int(part_test * n_data)
	
	shuffle_idx = n_buffer + n_train + jax.random.permutation(key, n_val + n_test)
	data_train = data_points[n_buffer:n_buffer+n_train]
	data_val = data_points[shuffle_idx[:n_val]]
	data_test = data_points[shuffle_idx[n_val:]]
	
	# unit scale
	data_scaler = MinMaxScaler(feature_range=(0, 1))
	data_scaler.fit(data_train)
	data_train = data_scaler.transform(data_train)
	data_val = data_scaler.transform(data_val)
	data_test = data_scaler.transform(data_test)
	
	# supervised split
	train_x, train_y = data_train[:,:-1], data_train[:,-1] # xyt, z
	val_x, val_y = data_val[:,:-1], data_val[:,-1]
	test_x, test_y = data_test[:,:-1], data_test[:,-1]
	
	# determine batch counts
	train_steps = int(jnp.ceil(train_x.shape[0] / batch_size))
	val_steps = int(jnp.ceil(val_x.shape[0] / batch_size))
	test_steps = int(jnp.ceil(test_x.shape[0] / batch_size))
	
	return (train_x, train_y, train_steps), (val_x, val_y, val_steps), (test_x, test_y, test_steps), data_scaler
