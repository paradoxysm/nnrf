import numpy as np

from nnrf.utils import create_random_state

class BatchDataset:
	"""
	Batch Dataset stores data and labels with capacity
	to shuffle, repeat, and batch data in a manner
	similar to Tensorflow's Dataset implementation.

	Parameters
	----------
	X : array-like, shape=(n_samples, n_features)
		Data.

	Y : array-like, shape=(n_samples, n_labels), default=None
		Labels.

	weights : array-like, shape=(n_samples,), default=None
		Sample weights. If None, then samples are equally weighted.

	seed : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	Attributes
	----------
	available : ndarray
		List of the indices corresponding to available
		data to draw from.

	batch_size : int, range=[1, n_samples]
		Batch size.

	order : list
		Order of operations used internally.

	i : int
		Number of times data has been drawn from the BatchDataset.
	"""
	def __init__(self, X, Y=None, weights=None, seed=None):
		self.X = np.array(X)
		self.Y = np.array(Y) if Y is not None else None
		if weights is not None : self.weights = np.array(weights)
		else : self.weights = np.ones(len(X))
		self.seed = create_random_state(seed=seed)
		self.available = np.arange(len(self.X)).astype(int)
		self.batch_size = 1
		self.n_batches = self._calculate_n_batches()
		self.order = []
		self.i = 0

	def batch(self, batch_size):
		"""
		Setup the BatchDataset to batch with the
		given batch size.

		Parameters
		----------
		batch_size :  int, range=[1, n_samples]
			Batch size.
		"""
		self.batch_size = batch_size
		self.n_batches = self._calculate_n_batches()
		self.order = [op for op in self.order if op != 'batch']
		self.order.append('batch')
		self.i = 0
		return self

	def repeat(self):
		"""
		Setup the BatchDataset to repeat the data.
		"""
		self.order = [op for op in self.order if op != 'repeat']
		self.order.append('repeat')
		self.i = 0
		return self

	def shuffle(self):
		"""
		Setup the BatchDataset to shuffle the data.
		"""
		self.order = [op for op in self.order if op != 'shuffle']
		self.order.append('shuffle')
		self.i = 0
		return self

	def next(self):
		"""
		Draw a batch from the BatchDataset.
		If this is the first batch, organize the dataset.
		If this batch would cause there to be less than another
		batch, reorganize the dataset.

		Returns
		-------
		next : ndarray or tuple of ndarray
			The batched data, and if available, labels and weights.
		"""
		if self.i == 0 : self.organize()
		if len(self.available) < self.batch_size:
			batch = self.available
			self.available = np.array([])
		else:
			batch = self.available[:self.batch_size]
			self.available = self.available[self.batch_size:]
		if len(self.available) < self.batch_size:
			self.organize(prepend=self.available)
		self.i += 1
		next = [self.X[batch]]
		if self.Y is None : next.append(None)
		else : next.append(self.Y[batch])
		next.append(self.weights[batch])
		return next

	def _calculate_n_batches(self):
		"""
		Calculate the number of batches that cover
		all data in the dataset.

		Returns
		-------
		n_batches : int
			Number of batches.
		"""
		return len(self.X) // self.batch_size + 1

	def organize(self, prepend=[], append=[]):
		"""
		Organize the BatchDataset according to `order`.
		In this manner, the order of shuffle, repeat, and
		batch affect how the data is drawn.

		Parameters
		----------
		prepend : list, default=[]
			Prepend these indices to the reorganized list.
			Data drawn will first exhaust this list.

		append : list, default=[]
			Append these indices to the reorganized list.
			Data drawn will exhaust this list last.
		"""
		order = np.array(self.order)
		try : shuffle = np.argwhere(order == 'shuffle').flatten()[0]
		except : shuffle = np.inf
		try : repeat = np.argwhere(order == 'repeat').flatten()[0]
		except : repeat = np.inf
		try : batch = np.argwhere(order == 'batch').flatten()[0]
		except : batch = np.inf
		length_Y = len(self.Y)
		if shuffle < repeat and shuffle < batch:
			self.available = np.arange(length_Y)
			self.seed.shuffle(self.available)
		elif repeat < shuffle:
			if shuffle < batch:
				self.available = self.seed.choice(np.arange(length_Y), length_Y)
			elif batch < shuffle and shuffle < np.inf:
				self.available = np.arange(length_Y).reshape(-1,self.batch_size)
				n_batches = np.arange(len(self.available))
				indices = self.seed.choice(n_batches, len(self.available))
				self.available = self.available[indices].flatten()
			else:
				self.available = np.arange(length_Y)
		elif batch < shuffle < repeat:
			self.available = np.arange(length_Y).reshape(-1,self.batch_size)
			self.seed.shuffle(self.available)
			self.available.flatten()
		self.available = np.concatenate((prepend, self.available, append), axis=0).astype(int)
