import numpy as np
from abc import ABC, abstractmethod

from nnrf._base import Base

def get_loss(name):
	"""
	Lookup table of default loss functions.

	Parameters
	----------
	name : LossFunction, None, str
		LossFunction to look up. Must be one of:
		 - 'mse' : Mean Squared Error.
		 - 'mae' : Mean Absolute Error.
		 - 'huber' : Huber Loss.
		 - 'hinge' : Hinge Loss.
		 - 'cross-entropy' : Crossentropy Loss.
		 - LossFunction : A custom implementation.
		 - None : Return None.
		Custom LossFunctions must implement `loss`, `gradient`
		functions and contain `scale` attribute.

	Returns
	-------
	loss : LossFunction or None
		The loss function.
	"""
	if name == 'mse' : return MSE()
	elif name == 'mae' : return MAE()
	elif name == 'huber' : return Huber()
	elif name == 'hinge' : return Hinge()
	elif name == 'cross-entropy' : return CrossEntropy()
	elif isinstance(name, (type(None), LossFunction)) : return name
	else : raise ValueError("Invalid loss function")


class LossFunction(Base, ABC):
	"""
	Base Loss Function.

	Attributes
	----------
	scale : tuple or None
		Acceptable range of loss function for gradient.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__()
		self.scale = None
		self.name = 'loss'

	@abstractmethod
	def loss(self, Y_hat, Y, *args, axis=1, **kwargs):
		"""
		Loss/error between labels `Y_hat` and targets `Y`.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Prediction labels.

		Y : array-like, shape=(n_samples, n_classes)
			Ground truth labels.

		axis : int, default=1
			Axis to compute loss.

		Returns
		-------
		loss : array-like, shape=(n_samples, n_classes)
			The error of each sample for each class.
		"""
		raise NotImplementedError("No loss function implemented")

	@abstractmethod
	def gradient(self, Y_hat, Y, *args, axis=1, **kwargs):
		"""
		Derivative of loss/error between labels `Y_hat` and targets `Y`.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Prediction labels.

		Y : array-like, shape=(n_samples, n_classes)
			Ground truth labels.

		axis : int, default=1
			Axis to compute loss.

		Returns
		-------
		dY : array-like, shape=(n_samples, n_classes)
			The derivative of the error of each sample for each class.
		"""
		raise NotImplementedError("No gradient function implemented")

class MSE(LossFunction):
	"""
	Mean Squared Error.
	Note that this implementation is really
	just a squared error and not a mean.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'mse'

	def loss(self, Y_hat, Y, axis=1):
		return np.square(Y - Y_hat)

	def gradient(self, Y_hat, Y, axis=1):
		return -2 * (Y - Y_hat)

class MAE(LossFunction):
	"""
	Mean Absolute Error.
	Note that this implementation is really
	just an absolute error and not a mean.
	"""
	def __init__(self):
		super().__init__()
		self.name = 'mae'

	def loss(self, Y_hat, Y, axis=1):
		return np.abs(Y - Y_hat)

	def gradient(self, Y_hat, Y, axis=1):
		grad = np.where(Y_hat > Y, 1, -1)
		grad[np.where(Y_hat == Y)] = 0
		return grad

class Huber(LossFunction):
	"""
	Huber Loss.

	Parameters
	----------
	delta : float, default=1
		Determines relationship between MSE and MAE
		character. Higher values approach MSE loss
		while lower values towards 0 approach MAE loss.
	"""
	def __init__(self, delta=1):
		super().__init__()
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
	"""
	Cross Entropy Loss.
	"""
	def __init__(self):
		super().__init__()
		self.scale = (0, 1)
		self.name = 'cross-entropy'

	def loss(self, Y_hat, Y, axis=1):
		return np.where(Y == 1, -np.log(Y_hat), -np.log(1 - Y_hat))

	def gradient(self, Y_hat, Y, axis=1):
		return - (np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

class Hinge(LossFunction):
	"""
	Hinge Loss.
	"""
	def __init__(self):
		super().__init__()
		self.scale = (-1, 1)
		self.name = 'hinge'

	def loss(self, Y_hat, Y, axis=1):
		return np.maximum(0, 1 - Y_hat * Y)

	def gradient(self, Y_hat, Y, axis=1):
		return np.where(1 - Y_hat * Y > 0, - Y, 0)
