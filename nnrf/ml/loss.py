import numpy as np
from abc import ABC, abstractmethod

from sleep.utils.data import normalize as n
from sleep.utils.misc import one_hot

def get_loss(name):
	if name == 'mse' : return MSE()
	elif name == 'mae' : return MAE()
	elif name == 'huber' : return Huber()
	elif name == 'hinge' : return Hinge()
	elif name == 'cross-entropy' : return CrossEntropy()
	elif isinstance(name, (None, LossFunction)) : return name
	else : raise ValueError("Invalid loss function")


class LossFunction(ABC):
	def __init__(self, *args, **kwargs):
		self.scale = None
		self.name = 'loss'

	@abstractmethod
	def loss(self, Y_hat, Y, *args, axis=1, **kwargs):
		raise NotImplementedError("No loss function implemented")

	@abstractmethod
	def gradient(self, Y_hat, Y, *args, axis=1, **kwargs):
		raise NotImplementedError("No gradient function implemented")

class MSE(LossFunction):
	def __init__(self):
		self.scale = None
		self.name = 'mse'

	def loss(self, Y_hat, Y, axis=1):
		return np.square(Y - Y_hat)

	def gradient(self, Y_hat, Y, axis=1):
		return -2 * (Y - Y_hat)

class MAE(LossFunction):
	def __init__(self):
		self.scale = None
		self.name = 'mae'

	def loss(self, Y_hat, Y, axis=1):
		return np.abs(Y - Y_hat)

	def gradient(self, Y_hat, Y, axis=1):
		grad = np.where(Y_hat > Y, 1, -1)
		grad[np.where(Y_hat == Y)] = 0
		return grad

class Huber(LossFunction):
	def __init__(self, delta=1):
		self.delta = delta
		self.scale = None
		self.name = 'huber'

	def loss(self, Y_hat, Y, axis=1):
		mask = np.abs(Y - Y_hat) > self.delta
		return np.where(mask, self.delta * (np.abs(Y - Y_hat) - 0.5 * self.delta),
							0.5 * np.square(Y - Y_hat))

	def gradient(self, Y_hat, Y, axis=1):
		mask = np.abs(Y - Y_hat) > self.delta
		grad = np.where(mask, None, Y_hat - Y)
		lin = np.where(Y_hat > Y, 1, -1)
		lin[np.where(Y_hat == Y)] = 0
		return np.where(mask, lin, Y_hat - Y)

class CrossEntropy(LossFunction):
	def __init__(self):
		self.scale = (0, 1)
		self.name = 'cross-entropy'

	def loss(self, Y_hat, Y, axis=1):
		return np.where(Y == 1, -np.log(Y_hat), -np.log(1 - Y_hat))

	def gradient(self, Y_hat, Y, axis=1):
		return - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

class Hinge(LossFunction):
	def __init__(self):
		self.scale = (-1, 1)
		self.name = 'hinge'

	def loss(self, Y_hat, Y, axis=1):
		return np.maximum(0, 1 - Y_hat * Y)

	def gradient(self, Y_hat, Y, axis=1):
		return np.where(1 - Y_hat * Y > 0, - Y, 0)
