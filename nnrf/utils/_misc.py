import numpy as np

def create_random_state(seed=None):
	"""
	Create a RandomState.

	Parameters
	----------
	seed : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	Returns
	-------
	random_state : RandomState
		A RandomState object.
	"""
	if seed is None:
		return np.random.mtrand._rand
	elif isinstance(seed, (int, np.integer)):
		return np.random.RandomState(seed=seed)
	elif isinstance(seed, np.random.RandomState):
		return seed
	else:
		raise ValueError("Seed must be None, an int, or a Random State")

def one_hot(Y, cols=None):
	"""
	Convert `Y` into a one-hot encoding.

	Parameters
	----------
	Y : ndarray, shape=(n_samples,)
		Data to encode.

	cols : int or None, default=None
		Number of columns to encode.
		Must be at least the number of unique
		values in `Y`.

	Returns
	-------
	one_hot : ndarray, shape=(n_samples, n_cols)
		One-hot encoded data.
	"""
	if Y.shape[1] == cols and set(Y) == set([0,1]):
		return Y
	if Y.size > len(Y) : raise ValueError("Matrix is not 1D")
	Y_ = Y.reshape(-1)
	if cols is None : cols = Y_.max() + 1
	elif cols < len(np.unique(Y)) : raise ValueError("There are more classes than cols")
	if cols > 1:
		one_hot = np.zeros((len(Y), cols), dtype=int)
		one_hot[np.arange(len(Y)), Y_] = 1
		return one_hot
	else:
		Y = np.where(Y < cols, 0, 1)
		return Y.reshape(-1, 1)

def decode(Y):
	"""
	Decode one-hot encoded data
	into a 1-dimensional array.

	Parameters
	----------
	Y : ndarray, shape=(n_samples, n_cols)
		One-hot encoded data.

	Returns
	-------
	decode : ndarray, shape=(n_samples,)
		Decoded data.
	"""
	if Y.shape[0] == Y.size:
		return np.squeeze(Y)
	elif set(Y) != set([0,1]):
		raise ValueError("Y must be one-hot encoded data with ints.")
	return np.argmax(Y, axis=1)

def calculate_batch(self, batch_size, length):
	"""
	Calculate the batch size for the data of given length.

	Parameters
	----------
	batch_size : int, float, default=None
		Batch size for training. Must be one of:
		 - int : Use `batch_size`.
		 - float : Use `batch_size * n_samples`.
		 - None : Use `n_samples`.

	length : int
		Length of the data to be batched.

	Returns
	-------
	batch_size : int
		Batch size.
	"""
	if batch_size is None : return length
	elif isinstance(batch_size, int) and batch_size > 0 and \
			batch_size <= length:
		return batch_size
	elif isinstance(batch_size, float) and 0 < batch_size <= 1:
		return int(batch_size * length)
	else:
		raise ValueError("Batch size must be None, an int less than %d," % length,
							"or a float within (0,1]")

def calculate_weight(self, Y, n_classes, class_weight=None, weights=None):
	"""
	Calculate the weights applied to the predicted labels,
	combining class weights and sample weights.

	Parameters
	----------
	Y : array-like, shape=(n_samples,)
		Target labels as integers.

	n_classes : int
		Number of classes.

	class_weight : dict, 'balanced', or None, default=None
		Weights associated with classes in the form
		`{class_label: weight}`. Must be one of:
		 - None : All classes have a weight of one.
		 - 'balanced': Class weights are automatically calculated as
						`n_samples / (n_samples * np.bincount(Y))`.

	weights : array-like, shape=(n_samples,), default=None
		Sample weights. If None, then samples are equally weighted.

	Returns
	-------
	weights : array-like, shape=(n_samples,)
		Weights combining sample weights and class weights.
	"""
	if class_weight is None and weights is None : return np.ones(len(Y))
	elif weights is None : weights = np.ones(len(Y))
	d = class_weight
	if isinstance(d, str) and d == 'balanced':
		l = len(Y) / (n_classes * np.bincount(Y))
		d = {k: l[k] for k in range(len(l))}
	elif isinstance(d, dict):
		k = list(d.keys())
		class_weights = np.where(Y == k, class_weight[k])
	elif d is None : class_weights = np.ones(len(Y))
	else : raise ValueError("Class Weight must either be a dict or 'balanced' or None")
	return weights * class_weights

def calculate_bootstrap(self, bootstrap_size, length):
	"""
	Calculate the bootstrap size for the data of given length.

	Parameters
	----------
	bootstrap_size : int, float, default=None
		Bootstrap size for training. Must be one of:
		 - int : Use `bootstrap_size`.
		 - float : Use `bootstrap_size * n_samples`.
		 - None : Use `n_samples`.

	length : int
		Length of the data to be bootstrapped.

	Returns
	-------
	bootstrap_size : int
		Bootstrap size.
	"""
	if bootstrap_size is None:
		bootstrap = length
	elif isinstance(bootstrap_size, int) and bootstrap > 0:
		bootstrap = bootstrap_size
	elif isinstance(bootstrap_size, float) and 0 < bootstrap_size <= 1:
		bootstrap = int(bootstrap_size * length)
	else : raise ValueError("Bootstrap Size must be None, a positive int or float in (0,1]")
