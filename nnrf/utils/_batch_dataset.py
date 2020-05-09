import numpy as np

from nnrf.utils import create_random_state

class BatchDataset:
	def __init__(self, X, Y=None, seed=None):
		self.X = np.array(X)
		self.Y = np.array(Y) if Y is not None else None
		self.seed = create_random_state(seed=seed)
		self.available = np.arange(len(self.X))
		self.batch_size = 1
		self.order = []
		self.i = 0

	def batch(self, batch_size):
		self.batch_size = batch_size
		self.order = [op for op in self.order if op != 'batch']
		self.order.append('batch')
		self.i = 0

	def repeat(self):
		self.order = [op for op in self.order if op != 'repeat']
		self.order.append('repeat')
		self.i = 0

	def shuffle(self):
		self.order = [op for op in self.order if op != 'shuffle']
		self.order.append('shuffle')
		self.i = 0

	def next(self):
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
		next = self.X[batch] if self.Y is None else (self.X[batch], self.Y[batch])
		return next

	def _organize(prepend=[], append=[]):
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
