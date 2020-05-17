import numpy as np
from abc import ABC, abstractmethod

from nnrf.utils._base import Base

def get_activation(name):
	"""
	Lookup table of default activation functions.

	Parameters
	----------
	name : Activation, None, str
		Activation to look up. Must be one of:
		 - 'linear' : Linear.
		 - 'binary' : Binary.
		 - 'sigmoid' : Sigmoid.
		 - 'tanh' : Tanh.
		 - 'arctan' : Arctan.
		 - 'relu' : Rectified Linear Unit (ReLU).
		 - 'prelu' : Parametric/Leaky ReLU.
		 - 'elu' : Exponential Linear Unit (ELU).
		 - 'noisy-relu' : Noisy ReLU.
		 - 'softmax' : Softmax.
		 - Activation : A custom implementation.
		 - None : Return None.
		Custom Activations must implement `activation`, `gradient`,
		`classify`, and `scale` functions.

	Returns
	-------
	activation : Activation or None
		The activation function.
	"""
	if name == 'linear' : return Linear()
	elif name == 'binary' : return Binary()
	elif name == 'sigmoid' : return Sigmoid()
	elif name == 'tanh' : return Tanh()
	elif name == 'arctan' : return Arctan()
	elif name == 'relu' : return ReLU()
	elif name == 'prelu' : return PReLU()
	elif name == 'elu' : return ELU()
	elif name == 'noisy-relu' : return NoisyReLU()
	elif name == 'softmax' : return Softmax()
	elif isinstance(name, (Activation, type(None))) : return name
	else : raise ValueError("Invalid activation function")

class Activation(Base, ABC):
	"""
	Base Activation class.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.name = 'activation'

	@abstractmethod
	def activation(self, X, *args, **kwargs):
		"""
		Activation function. Returns `X` transformed
		by the activation function.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data.

		Returns
		-------
		A : array-like, shape=(n_samples, n_features)
			Data transformed by activation function.
		"""
		raise NotImplementedError("No activation function implemented")

	@abstractmethod
	def gradient(self, X, *args, **kwargs):
		"""
		Derivative of activation function. Returns gradient
		of the activation function at `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data.

		Returns
		-------
		dA : array-like, shape=(n_samples, n_features)
			Gradient of activation function at `X`.
		"""
		raise NotImplementedError("No gradient function implemented")

	def scale(self, Y, scale, *args, **kwargs):
		"""
		Scale `Y` to the scale given by the LossFunction
		`loss`.

		Parameters
		----------
		Y : array-like, shape=(n_samples, n_features)
			Data.

		scale : tuple, shape=(2,)
			Interval to scale data into.

		Returns
		-------
		S : array-like, shape=(n_samples, n_features)
			Scaled data.
		"""
		return Y

class Linear(Activation):
	"""
	Linear Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'linear'

	def activation(self, X):
		return X

	def gradient(self, X):
		return np.ones(X.shape)

class Binary(Activation):
	"""
	Binary Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'binary'

	def activation(self, X):
		return np.where(X < 0, 0, 1)

	def gradient(self, X):
		return np.zeros(X.shape)

	def scale(self, Y, scale):
		return np.where(Y == 0, scale[0], scale[1])

class Sigmoid(Activation):
	"""
	Sigmoid Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'sigmoid'

	def activation(self, X):
		return 1 / (1 + np.exp(-X))

	def gradient(self, X):
		sig = self.activation(X)
		return sig * (1 - sig)

	def scale(self, Y, scale):
		size = scale[1] - scale[0]
		return size * Y + scale[0]

class Tanh(Activation):
	"""
	Tanh Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'tanh'

	def activation(self, X):
		return 2 / (1 + np.exp(-2 * X)) - 1

	def gradient(self, X):
		return 1 - np.square(self.activation(X))

	def scale(self, Y, scale):
		size = (scale[1] - scale[0]) / 2
		return size * Y + scale[0]

class Arctan(Activation):
	"""
	Arctan Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'arctan'

	def activation(self, X):
		return np.arctan(X)

	def gradient(self, X):
		return 1 / (np.square(X) + 1)

	def scale(self, Y, scale):
		size = (scale[1] - scale[0]) / np.pi
		return size * Y + scale[0]

class ReLU(Activation):
	"""
	ReLU Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'relu'

	def activation(self, X):
		return np.maximum(0, X)

	def gradient(self, X):
		return np.where(X > 0, 1, 0)

	def scale(self, Y, scale):
		Y = Y + scale[0]
		Y = np.where(Y > scale[1], scale[1], Y)
		return np.where(Y < scale[0], scale[0], Y)

class PReLU(ReLU):
	"""
	PReLU Activation Function.

	Parameters
	----------
	a : float, default=0.01
		Leakiness of ReLU.
	"""
	def __init__(self, a=0.01):
		super().__init__()
		self.name = 'prelu'
		self.a = a

	def activation(self, X):
		return np.where(X < 0, self.a * X, X)

	def gradient(self, X):
		return np.where(X > 0, 1, self.a)

class ELU(ReLU):
	"""
	ELU Activation Function.

	Parameters
	----------
	a : float, default=0.1
		Factor of exponential function.
	"""
	def __init__(self, a=0.1):
		if a < 0 : raise ValueError("Hyperparameter must be non-negative")
		super().__init__()
		self.a = a
		self.name = 'elu'

	def activation(self, X):
		return np.where(X < 0, self.a * (np.exp(X) - 1), X)

	def gradient(self, X):
		return np.where(X > 0, 1, np.exp(X))

class NoisyReLU(ReLU):
	"""
	Noisy ReLU Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'noisy-relu'

	def activation(self, X):
		sigma = np.std(X, axis=axis)
		return np.maximum(0, X + np.random.normal(scale=sigma))

class Softmax(Activation):
	"""
	Softmax Activation Function.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'softmax'

	def activation(self, X, axis=1):
		exp = np.exp(X)
		return exp / np.sum(exp, axis=axis).reshape(-1,1)

	def gradient(self, X, axis=1):
		s = self.activation(X, axis=axis)
		return s * (1 - s)

	def scale(self, Y, scale):
		size = scale[1] - scale[0]
		return size * Y + scale[0]
