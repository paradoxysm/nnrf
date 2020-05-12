import numpy as np

from nnrf.ml import get_loss, get_activation, get_regularizer
from nnrf.utils import check_XY, one_hot, decode, calculate_weight, \
						create_random_state, BatchDataset
from nnrf.analysis import get_metrics

from nnrf._base import BaseClassifier

class NeuralNetwork(BaseClassifier):
	"""
	Basic Neural Network.

	Parameters
	----------
	layers : tuple, default=(100,)
		The ith element represents the number of
		neurons in the ith hidden layer.

	activation : str, Activation, default=PReLU(0.2)
		Activation function to use at each node in the NNDT.
		Must be one of the default loss functions or an
		object that extends Activation.

	loss : str, LossFunction, default='cross-entropy'
		Loss function to use for training. Must be
		one of the default loss functions or an object
		that extends LossFunction.

	alpha : float, default=0.001
		Learning rate. model uses gradient descent.

	batch_size : int, float, default=None
		Batch size for training. Must be one of:
		 - int : Use `batch_size`.
		 - float : Use `batch_size * n_samples`.
		 - None : Use `n_samples`.

	max_iter : int, default=10
		Maximum number of epochs to conduct during training.

	tol : float, default=1e-4
		Convergence criteria for early stopping.

	random_state : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	regularize : str, Regularizer, None, default=None
		Regularization function to use at each node in the model.
		Must be one of the default regularizers or an object that
		extends Regularizer. If None, no regularization is done.

	class_weight : dict, 'balanced', or None, default=None
		Weights associated with classes in the form
		`{class_label: weight}`. Must be one of:
		 - None : All classes have a weight of one.
		 - 'balanced': Class weights are automatically calculated as
						`n_samples / (n_samples * np.bincount(Y))`.

	verbose : int, default=0
		Verbosity of estimator; higher values result in
		more verbose output.

	warm_start : bool, default=False
		Determines warm starting to allow training to pick
		up from previous training sessions.

	metric : str, Metric, or None, default='accuracy'
		Metric for estimator score.

	Attributes
	----------
	n_layers : int
		Number of hidden/output layers in the model.

	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.

	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.

	weights_ : list of ndarray, shape=(layers,)
		Weights of the model.

	bias_ : list of ndarray, shape=(layers,)
		Biases of the model.
	"""
	def __init__(self, layers=(100,), activation='relu', loss='cross-entropy',
					alpha=0.001, batch_size=None, max_iter=10, tol=1e-4,
					random_state=None, regularize=None, class_weight=None,
					metric='accuracy', verbose=0, warm_start=False):
		super().__init__(batch_size=batch_size, verbose=verbose,
						warm_start=warm_start, class_weight=class_weight,
						metric=metric, random_state=random_state)
		self.activation = get_activation(activation)
		self.softmax = get_activation('softmax')
		self.regularizer = get_regularizer(regularize)
		self.loss = get_loss(loss)
		self.alpha = alpha
		self.layers = layers
		self.n_layers_ = len(self.layers)
		self.weights_ = []
		self.bias_ = []
		self.n_classes_ = None
		self.n_features_ = None

	def _initialize(self):
		"""
		Initialize the parameters of the neural network.
		"""
		if self.layers == tuple():
			self.weights_ = [self.random_state.randn((self.n_features_, self.n_classes_)) * 0.1]
			self.bias_ = [self.random_state.randn((self.n_classes_)) * 0.1]
			self.n_layers_ = 1
			self.layers = (self.n_features,)
			return
		self.weights_.append(self.random_state.randn((self.n_features_, self.layers[0])) * 0.1)
		self.bias_.append(self.random_state.randn(layers[0]))
		for l in range(self.n_layers_ - 1):
			self.weights_.append(self.random_state.randn((self.layers[l], self.layers[l+1])) * 0.1)
			self.bias_.append(self.random_state.randn(self.layers[l]))
		self.weights_.append(self.random_state.randn((self.layers[-1], self.n_classes_)) * 0.1)
		self.bias_.append(self.random_state.randn(self.n_classes_) * 0.1)

	def _is_fitted(self):
		"""
		Return True if the model is properly ready
		for prediction.

		Returns
		-------
		fitted : bool
			True if the model can be used to predict data.
		"""
		weights = len(self.weights_) > 0
		bias = len(self.bias_) > 0
		return weights and bias and self.fitted_

	def _forward(self, X):
		"""
		Conduct the forward propagation steps through the neural
		network.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Output.
		"""
		self._x, self._z = [], []
		X_ = X
		for l in range(self.n_layers):
			Z = np.dot(self.weights_[l], X_) + self.bias_[l]
			if l < self.n_layers - 1 : A = self.activation.activation(Z)
			else : A = self.softmax.activation(Z)
			self._z.append(Z)
			self._x.append(X_)
			X_ = A
		return A

	def _backward(self, Y_hat, Y, weights=None):
		"""
		Conduct the backward propagation steps through the
		neural network.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Model output.

		Y : array-like, shape=(n_samples, n_classes)
			Target labels in one hot encoding.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.
		"""
		weights = calculate_weight(decode(Y), self.n_classes_,
					class_weight=self.class_weight, weights=weights)
		m = len(Y)
		dY = self.loss.gradient(Y_hat, Y) * weights.reshape(-1,1)
		for l in range(self.n_layers - 1, -1, -1):
			if l == self.n_layers - 1:
				dZ = dY * self.softmax.gradient(self._z[-1])
			else : dZ = dY * self.activation.gradient(self._z[l])
			dW = np.dot(self._x[l], dZ) / m
			db = np.sum(dZ, axis=0) / m
			if self.regularizer is not None:
				self.weights_[l] -= self.regularizer.gradient(self.weights_[l])
			self.weights_[l] -= self.alpha * dW
			self.bias_[l] -= self.alpha * db
			dY = np.dot(dZ, self.weights_[l].T)
