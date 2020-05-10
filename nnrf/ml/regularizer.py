import numpy as np
from abc import ABC, abstractmethod

from nnrf._base import Base

def get_regularizer(name):
	if name == 'l1' : return L1()
	elif name == 'l2' : return L2()
	elif name == 'l1-l2' : return L1L2()
	elif isinstance(name, (None, Regularizer)) : return name
	else : raise ValueError("Invalid regularizer")

class Regularizer(Base, ABC):
	def __init__(self, *args, **kwargs):
		self.name = 'regularizer'

	def cost(self, *args, **kwargs):
		raise NotImplementedError("No cost function implemented")

	def gradient(self, *args, **kwargs):
		raise NotImplementedError("No gradient function implemented")

class L1(Regularizer):
	def __init__(self, c=0.01):
		self.name = 'l1'
		self.c = c

	def cost(self, w, axis=1):
		return self.c * np.linalg.norm(w, 1, axis=axis)

	def gradient(self, w, axis=1):
		return self.c * w

class L2(Regularizer):
	def __init__(self, c=0.01):
		self.name = 'l2'
		self.c = c

	def cost(self, w, axis=1):
		return 0.5 * self.c * np.linalg.norm(w, axis=axis)

	def gradient(self, w, axis=1):
		return self.c * w

class L1L2(Regularizer):
	def __init__(self, l1=0.01, l2=0.01, weight=0.5):
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
	if name == 'unit' : return UnitNorm
	elif name == 'maxnorm' : return MaxNorm
	elif name == 'minmax' : return MinMaxNorm
	elif isinstance(name, (None, Constraint)) : return name
	else : raise ValueError("Invalid regularizer")

class Constraint(Base, ABC):
	def __init__(self, *args, **kwargs):
		self.name = 'constraint'

	def constrain(self, *args, **kwargs):
		raise NotImplementedError("No constrain function implemented")

class UnitNorm(Constraint):
	def __init__(self, c, norm='l2'):
		self.name = 'unitnorm'
		self.c = c
		self.norm = get_regularizer('l2')(c=1)

	def constraint(self, w, axis=0):
		w_norm = self.norm.cost(w, axis=axis)
		return w / w_norm

class MaxNorm(Constraint):
	def __init__(self, c=4, norm='l2'):
		self.name = 'maxnorm'
		self.c = c
		self.norm = get_regularizer('l2')(c=1)

	def constraint(self, w, axis=0):
		w_norm = self.norm.cost(w, axis=axis)
		w_norm = np.where(w_norm > self.c, w_norm / self.c, 1)
		return w / w_norm

class MinMaxNorm(Constraint):
	def __init__(self, min=0, max=4, norm='l2'):
		self.name = 'minmax'
		self.min = min
		self.max = max
		self.norm = get_regularizer('l2')(c=1)

	def constraint(self, w, axis=0):
		w_norm = self.norm.cost(w, axis=axis)
		max = np.where(w_norm > self.max)
		min = np.where(w_norm < self.min)
		norm = np.ones(w.shape)
		norm[max] = w_norm[max] / self.max
		norm[min] = w_norm[min] / self.min
		return w / norm
