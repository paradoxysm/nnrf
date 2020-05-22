import numpy as np
from abc import ABC, abstractmethod

from nnrf.utils._base import Base

def get_optimizer(name):
	"""
	Lookup table of default optimizers.

	Parameters
	----------
	name : Optimizer, None, str
		Optimizer to look up. Must be one of:
		 - 'sgd' : Gradient Descent.
		 - 'momentum' : Gradient Descent with Momentum.
		 - 'rmsprop' : RMSProp.
		 - 'adam' : Adam.
		 - Regularizer : A custom implementation.
		 - None : Return None.
		Custom Optimizer must implement `update`
		functions.

	Returns
	-------
	optimizer : Optimizer or None
		The optimizer.
	"""
	if name == 'sgd' : return SGD()
	elif name == 'momentum' : return Momentum()
	elif name == 'rmsprop' : return RMSProp()
	elif name == 'adam' : return Adam()
	elif isinstance(name, (type(None), Optimizer)) : return name
	else : raise ValueError("Invalid optimizer")

class Optimizer(Base, ABC):
	"""
	Base Optimizer
	"""
	def __init__(self, *args, **kwargs):
		self.name = 'optimizer'
		self.memory = {}

	def setup(self, keys, *args, **kwargs):
		"""
		Setup the memory cache.

		Parameters
		----------
		keys : list
			List of key names for the cache.
		"""
		return

	def update(self, key, gradient, *args, **kwargs):
		"""
		Calculate the parameter update for a given
		parameter.

		Parameters
		----------
		key : str
			Name of the key as stored in the memory cache.

		gradient : float, array-like
			Gradient of the respective parameter.

		Returns
		-------
		update : float, array-like
			Update to be applied.
		"""
		if key not in self.memory.keys():
			raise ValueError("Optimizer not setup to handle these gradients")

	def _update_mem(self, key, d, beta):
		"""
		Update cache with momentum.

		Parameters
		----------
		key : str
			Name of the key as stored in the memory cache.

		d : float, array-like
			Current value.

		beta : float
			Momentum.
		"""
		if self.memory[key] is None : self.memory[key] = d
		else : self.memory[key] = beta * self.memory[key] + d

class SGD(Optimizer):
	"""
	Gradient Descent.

	Parameters
	----------
	alpha : float, default=0.001
		Learning Rate.
	"""
	def __init__(self, alpha=0.001):
		super().__init__()
		self.alpha = alpha
		self.name = 'sgd'

	def update(self, key, gradient):
		return self.alpha * gradient


class Momentum(Optimizer):
	"""
	Gradient Descent with Momentum.

	Parameters
	----------
	alpha : float, default=0.001
		Learning Rate.

	beta : float, default=0.9
		Momentum for past gradients.
	"""
	def __init__(self, alpha=0.001, beta=0.9):
		super().__init__()
		self.alpha = alpha
		self.beta = beta
		self.name = 'sgd'

	def setup(self, keys):
		for k in keys:
			self.memory['M'+k] = None

	def update(self, key, gradient):
		key = 'M' + key
		super().update(key, gradient)
		d = self.alpha + gradient
		self._update_mem(key, d, self.beta)
		return self.memory[key]


class RMSProp(Optimizer):
	"""
	RMSProp.

	Parameters
	----------
	alpha : float, default=0.001
		Learning Rate.

	beta : float, default=0.9
		Momentum for past squared gradients.
	"""
	def __init__(self, alpha=0.001, beta=0.9):
		super().__init__()
		self.alpha = alpha
		self.beta = beta
		self.name = 'rmsprop'

	def setup(self, keys):
		for k in keys:
			self.memory['E'+k] = None

	def update(self, key, gradient):
		key = 'E' + key
		super().update(key, gradient)
		d = (1 - self.beta) * self.square(gradient)
		self._update_mem(key, d, self.beta)
		return self.alpha / np.sqrt(self.memory[key] + 1e-8) * gradient


class Adam(Optimizer):
	"""
	Adam.

	Parameters
	----------
	alpha : float, default=0.001
		Learning Rate.

	beta1 : float, default=0.9
		Momentum for past gradients.

	beta2 : float, default=0.999
		Momentum for past squared gradients.
	"""
	def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999):
		super().__init__()
		self.alpha = alpha
		self.beta1 = beta1
		self.beta2 = beta2
		self.name = 'adam'

	def setup(self, keys):
		for k in keys:
			self.memory['M'+k] = None
			self.memory['V'+k] = None
			self.memory['t_m'] = 0
			self.memory['t_v'] = 0

	def update(self, key, gradient):
		m_key = 'M' + key
		v_key = 'V' + key
		super().update(m_key, gradient)
		super().update(v_key, gradient)
		M = (1 - self.beta1) * gradient
		V = (1 - self.beta2) * np.square(gradient)
		self._update_mem(m_key, M, self.beta1)
		self._update_mem(v_key, V, self.beta2)
		M = self._bias_correct(m_key, 't_m', self.beta1)
		V = self._bias_correct(v_key, 't_v', self.beta2)
		return self.alpha / (np.sqrt(V) + 1e-8) * M

	def _bias_correct(self, key, t, beta):
		self.memory[t] += 1
		return self.memory[key] / (1 - np.power(beta, self.memory[t]))
