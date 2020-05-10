import numpy as np
from abc import ABC, abstractmethod

from nnrf._base import Base

def get_activation(name):
	if name == 'linear' : return Linear()
	elif name == 'exponential' : return Exponential()
	elif name == 'binary' : return Binary()
	elif name == 'sigmoid' : return Sigmoid()
	elif name == 'tanh' : return Tanh()
	elif name == 'arctan' : return Arctan()
	elif name == 'relu' : return ReLU()
	elif name == 'softplus' : return Softplus()
	elif name == 'prelu' : return PReLU()
	elif name == 'leaky_relu' : return LeakyReLU()
	elif name == 'elu' : return ELU()
	elif name == 'noisy_relu' : return NoisyReLU()
	elif name == 'softmax' : return Softmax()
	elif isinstance(name, (Activation, None)) : return name
	else : raise ValueError("Invalid activation function")

class Activation(Base, ABC):
	def __init__(self, *args, **kwargs):
		self.name = 'activation'

	@abstractmethod
	def activation(self, X, *args, **kwargs):
		raise NotImplementedError("No activation function implemented")

	@abstractmethod
	def gradient(self, X, *args, **kwargs):
		raise NotImplementedError("No gradient function implemented")

	def classify(self, X, *args, **kwargs):
		return np.where(X > 0, 1, 0)

	def scale(self, Y, loss, *args, **kwargs):
		return Y

class Linear(Activation):
	def __init__(self):
		self.name = 'linear'

	def activation(self, X):
		return X

	def gradient(self, X):
		return np.ones(X.shape)

class Binary(Activation):
	def __init__(self):
		self.name = 'binary'

	def activation(self, X):
		return np.where(X < 0, 0, 1)

	def gradient(self, X):
		if 0 in X : raise Warning("0 encountered in data, resulting in NaN.")
		return np.where(X != 0, 0, np.nan)

	def classify(self, X):
		return X

	def scale(self, Y, loss):
		range = loss.scale
		if range is not None:
			return np.where(Y == 0, range[0], range[1])
		return Y

class Sigmoid(Activation):
	def __init__(self):
		self.name = 'sigmoid'

	def activation(self, X):
		return 1 / (1 + np.exp(-X))

	def gradient(self, X):
		sig = sigmoid(X, axis=axis)
		return sig * (1 - sig)

	def classify(self, X):
		return np.where(X < 0.5, 0, 1)

	def scale(self, Y, loss):
		range = loss.scale
		if range is not None:
			size = range[1] - range[0]
			return size * Y + range[0]
		return Y

class Tanh(Activation):
	def __init__(self):
		self.name = 'tanh'

	def activation(self, X):
		return 2 / (1 + np.exp(-2 * X)) - 1

	def gradient(self, X):
		return 1 - np.square(tanh(X, axis=axis))

	def classify(self, X):
		raise Warning("Tanh should not be used for classification")
		return X

	def scale(self, Y, loss):
		range = loss.scale
		if range is not None:
			size = (range[1] - range[0]) / 2
			return size * Y + range[0]
		return Y

class Arctan(Activation):
	def __init__(self):
		self.name = 'arctan'

	def activation(self, X):
		return np.arctan(X)

	def gradient(self, X):
		return 1 / (np.square(X) + 1)

	def classify(self, X):
		raise Warning("Arctan should not be used for classification")
		return X

	def scale(self, Y, loss):
		range = loss.scale
		if range is not None:
			size = (range[1] - range[0]) / np.pi
			return size * Y + range[0]
		return Y

class Softplus(Activation):
	def __init__(self):
		self.name = 'softplus'

	def activation(self, X):
		return np.log(1 + np.exp(X))

	def gradient(self, X):
		return 1 / (1 + np.exp(-X))

	def classify(self, X):
		raise Warning("Softplus should not be used for classification")
		return X

	def scale(self, Y, loss):
		raise Warning("Softplus should not be used for scaling")
		return Y

class ReLU(Activation):
	def __init__(self):
		self.name = 'relu'

	def activation(self, X):
		return np.maximum(0, X)

	def gradient(self, X):
		return np.where(X > 0, 1, 0)

	def scale(self, Y, loss):
		range = loss.scale
		if range is not None:
			Y = Y + range[0]
			Y = np.where(Y > range[1], range[1], Y)
			return np.where(Y < range[0], range[0], Y)
		return Y

class PReLU(ReLU):
	def __init__(self, a=0.01):
		self.name = 'prelu'
		self.a = a

	def activation(self, X):
		return np.where(X < 0, self.a * X, X)

	def gradient(self, X):
		return np.where(X > 0, 1, self.a)

	def classify(self, X):
		raise Warning("PReLU should not be used for classification, using ReLU instead")
		return np.where(X > 0, 1, 0)

class LeakyReLU(PReLU):
	def __init__(self):
		super().__init__(0.01)
		self.name = 'leaky_relu'

	def classify(self, X):
		raise Warning("LeakyReLU should not be used for classification, using ReLU instead")
		return np.where(X > 0, 1, 0)

class ELU(ReLU):
	def __init__(self, a=0.1):
		if a < 0 : raise ValueError("Hyperparameter must be non-negative")
		self.a = a
		self.name = 'elu'

	def activation(self, X):
		return np.where(X < 0, self.a * (np.exp(X) - 1), X)

	def gradient(self, X):
		return np.where(X > 0, 1, elu(X, self.a, axis=axis))

	def classify(self, X):
		raise Warning("LeakyReLU should not be used for classification, using ReLU instead")
		return np.where(X > 0, 1, 0)

class NoisyReLU(ReLU):
	def __init__(self):
		self.name = 'noisy_relu'

	def activation(self, X):
		sigma = np.std(X, axis=axis)
		return np.maximum(0, X + np.random.normal(scale=sigma))

class Softmax(Activation):
	def __init__(self):
		self.name = 'softmax'

	def activation(self, X, axis=1):
		exp = np.exp(X)
		return exp / np.sum(exp, axis=axis).reshape(-1,1)

	def gradient(self, X, axis=1):
		s = self.activation(X, axis=axis)
		return s * (1 - s)

	def classify(self, X, axis=1):
		return np.argmax(X, axis=axis)

	def scale(self, Y, loss):
		range = loss.scale
		if range is not None:
			size = range[1] - range[0]
			return size * Y + range[0]
		return Y
