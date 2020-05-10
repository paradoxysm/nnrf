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
	def __init__(self, X, Y=None, seed=None):
		self.X = np.array(X)
		self.Y = np.array(Y) if Y is not None else None
		self.seed = create_random_state(seed=seed)
		self.available = np.arange(len(self.X))
		self.batch_size = 1
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
		self.order = [op for op in self.order if op != 'batch']
		self.order.append('batch')
		self.i = 0

	def repeat(self):
		"""
		Setup the BatchDataset to repeat the data.
		"""
		self.order = [op for op in self.order if op != 'repeat']
		self.order.append('repeat')
		self.i = 0

	def shuffle(self):
		"""
		Setup the BatchDataset to shuffle the data.
		"""
		self.order = [op for op in self.order if op != 'shuffle']
		self.order.append('shuffle')
		self.i = 0

	def next(self):
		"""
		Draw a batch from the BatchDataset.
		If this is the first batch, organize the dataset.
		If this batch would cause there to be less than another
		batch, reorganize the dataset.

		Returns
		-------
		next : ndarray or tuple of ndarray
			The batched data, and if available, labels.
		"""
		if self.i == 0 : self._organize()
		if len(self.available) <= 2 * self.batch_size:
			if len(self.available) < self.batch_size:
				return [], []
			batch = self.available[:batch_size]
			self.available = self.available[batch_size:]
			self._organize(prepend=self.available)
		else:
			batch = self.available[:batch_size]
			self.available = self.available[batch_size:]
		i += 1
		if self.Y is None:
			return self.X[batch]
		else:
			return (self.X[batch], self.Y[batch].reshape(-1))

	def _organize(prepend=[], append=[]):
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
		try : shuffle = np.argwhere(self.order == 'shuffle').flatten()[0]
		except : shuffle = np.inf
		try : repeat = np.argwhere(self.order == 'repeat').flatten()[0]
		except : repeat = np.inf
		try : batch = np.argwhere(self.order == 'batch').flatten()[0]
		except : batch = np.inf
		if shuffle < repeat and shuffle < batch:
			self.available = np.arange(len(X))
			self.seed.shuffle(self.available)
		elif repeat < shuffle:
			if shuffle < batch:
				self.available = self.seed.choice(np.arange(len(Y)), len(Y))
			elif batch < shuffle and shuffle < np.inf:
				self.available = np.arange(len(X)).reshape(-1,self.batch_size)
				n_batches = np.arange(len(self.available))
				indices = self.seed.choice(n_batches, len(self.available))
				self.available = self.available[indices].flatten()
			else:
				self.available = np.arange(len(X))
		elif batch < shuffle < repeat:
			self.available = np.arange(len(X)).reshape(-1,self.batch_size)
			self.seed.shuffle(self.available)
			self.available.flatten()
		self.available = np.concatenate((prepend, self.available, append), axis=0)
