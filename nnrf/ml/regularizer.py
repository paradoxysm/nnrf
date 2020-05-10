import numpy as np
from abc import ABC, abstractmethod

from nnrf._base import Base

def get_regularizer(name):
	"""
	Lookup table of default regularization functions.

	Parameters
	----------
	name : Regularizer, None, str
		Regularizer to look up. Must be one of:
		 - 'l1' : L1 weight-decay.
		 - 'l2' : L2 weight-decay.
		 - 'l1-l2' : Combined L1-L2 weight-decay.
		 - Regularizer : A custom implementation.
		 - None : Return None.
		Custom Regularizer must implement `cost`, `gradient`
		functions.

	Returns
	-------
	regularizer : Regularizer or None
		The regularization function.
	"""
	if name == 'l1' : return L1()
	elif name == 'l2' : return L2()
	elif name == 'l1-l2' : return L1L2()
	elif isinstance(name, (None, Regularizer)) : return name
	else : raise ValueError("Invalid regularizer")

class Regularizer(Base, ABC):
	"""
	Base Regularizer.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.name = 'regularizer'

	def cost(self, w, *args, axis=1, **kwargs):
		"""
		Calculate the cost penalty.

		Parameters
		----------
		w : ndarray
			Array of weights.

		axis : int, default=1
			Axis to compute cost.

		Returns
		-------
		cost : float
			Cost.
		"""
		raise NotImplementedError("No cost function implemented")

	def gradient(self, w, *args, axis=1 **kwargs):
		"""
		Calculate the gradient of the cost penalty.

		Parameters
		----------
		w : ndarray
			Array of weights.

		axis : int, default=1
			Axis to compute cost.

		Returns
		-------
		gradient : float
			Gradient.
		"""
		raise NotImplementedError("No gradient function implemented")

class L1(Regularizer):
	"""
	L1 Norm Regularizer.

	Parameters
	----------
	c : float, default=0.01
		Regularization parameter. Larger values
		result in larger penalties.
	"""
	def __init__(self, c=0.01):
		super().__init__()
		self.name = 'l1'
		self.c = c

	def cost(self, w, axis=1):
		return self.c * np.linalg.norm(w, 1, axis=axis)

	def gradient(self, w, axis=1):
		return self.c * w

class L2(Regularizer):
	"""
	L2 Norm Regularizer.

	Parameters
	----------
	c : float, default=0.01
		Regularization parameter. Larger values
		result in larger penalties.
	"""
	def __init__(self, c=0.01):
		super().__init__()
		self.name = 'l2'
		self.c = c

	def cost(self, w, axis=1):
		return 0.5 * self.c * np.linalg.norm(w, axis=axis)

	def gradient(self, w, axis=1):
		return self.c * w

class L1L2(Regularizer):
	"""
	L1-L2 Norm Regularizer.

	Parameters
	----------
	l1 : float, default=0.01
		Regularization parameter for L1 Norm. Larger values
		result in larger penalties.

	l2 : float, default=0.01
		Regularization parameter for L2 Norm. Larger values
		result in larger penalties.

	weight : float, range=[0,1], default=0.5
		Weight of L1 Norm compared to L2 Norm.
	"""
	def __init__(self, l1=0.01, l2=0.01, weight=0.5):
		super().__init__()
		self.name = 'l1-l2'
		self.l1 = L1(c=l1)
		self.l2 = L2(c=l2)
		self.weight = weight

	def cost(self, w, axis=1):
		return self.weight * self.l1.cost(w, axis=axis) + \
		 		(1 - self.weight) * self.l2.cost(w, axis=axis)

	def gradient(self, w, axis=1):
		return self.weight * self.l1.gradient(w, axis=axis) + \
				(1 - self.weight) * self.l2.gradient(w, axis=axis)



def get_constraint(name):
	"""
	Lookup table of default weight constraint functions.

	Parameters
	----------
	name : Constraint, None, str
		Constraint to look up. Must be one of:
		 - 'l1' : L1 weight-decay.
		 - 'l2' : L2 weight-decay.
		 - 'l1-l2' : Combined L1-L2 weight-decay.
		 - Constraint : A custom implementation.
		 - None : Return None.
		Custom Constraint must implement `constrain`
		function.

	Returns
	-------
	constraint : Constraint or None
		The constraint function.
	"""
	if name == 'unit' : return UnitNorm
	elif name == 'maxnorm' : return MaxNorm
	elif name == 'minmax' : return MinMaxNorm
	elif isinstance(name, (None, Constraint)) : return name
	else : raise ValueError("Invalid regularizer")

class Constraint(Base, ABC):
	"""
	Base Constraint Function.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.name = 'constraint'

	def constrain(self, w, *args, axis=1, **kwargs):
		"""
		Constrain weights.

		Parameters
		----------
		w : ndarray
			Array of weights.

		axis : int, default=1
			Axis to compute cost.

		Returns
		-------
		w : ndarray
			Constrained weights.
		"""
		raise NotImplementedError("No constrain function implemented")

class UnitNorm(Constraint):
	"""
	Unit Norm Constraint using L2 Norm.

	Parameters
	----------
	c : float, default=1
		Constant norm of the weights after constraint.
	"""
	def __init__(self, c=1, norm='l2'):
		super().__init__()
		self.name = 'unitnorm'
		self.c = c
		self.norm = get_regularizer('l2')(c=1)

	def constrain(self, w, axis=1):
		w_norm = self.norm.cost(w, axis=axis)
		return c * w / w_norm

class MaxNorm(Constraint):
	"""
	Max Norm Constraint using L2 Norm.

	Parameters
	----------
	c : float, default=4
		Max norm of the weights after constraint.
	"""
	def __init__(self, c=4, norm='l2'):
		super().__init__()
		self.name = 'maxnorm'
		self.c = c
		self.norm = get_regularizer('l2')(c=1)

	def constraint(self, w, axis=1):
		w_norm = self.norm.cost(w, axis=axis)
		w_norm = np.where(w_norm > self.c, w_norm / self.c, 1)
		return w / w_norm

class MinMaxNorm(Constraint):
	"""
	Min-Max Norm Constraint using L2 Norm.

	Parameters
	----------
	min : float, default=0
		Minimum norm of weights after constraint.

	max : float, default=4
		Maximum norm of weights after constraint.
	"""
	def __init__(self, min=0, max=4, norm='l2'):
		super().__init__()
		self.name = 'minmax'
		self.min = min
		self.max = max
		self.norm = get_regularizer('l2')(c=1)

	def constraint(self, w, axis=1):
		w_norm = self.norm.cost(w, axis=axis)
		max = np.where(w_norm > self.max)
		min = np.where(w_norm < self.min)
		norm = np.ones(w.shape)
		norm[max] = w_norm[max] / self.max
		norm[min] = w_norm[min] / self.min
		return w / norm
