import numpy as np

from nnrf.ml import get_activation, get_regularizer, get_optimizer
from nnrf.utils import check_XY, one_hot, decode, calculate_weight, \
						create_random_state, BatchDataset
from nnrf.analysis import get_metrics

from nnrf.utils._estimator import BaseClassifier

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

	optimizer : str, Optimizer, default='adam'
		Optimization method. Must be one of
		the default optimizers or an object that
		extends Optimizer.

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
					optimizer='adam', batch_size=None, max_iter=100, tol=1e-4,
					random_state=None, regularize=None, class_weight=None,
					metric='accuracy', verbose=0, warm_start=False):
		super().__init__(loss=loss, max_iter=max_iter, tol=tol,
						batch_size=batch_size, verbose=verbose,
						warm_start=warm_start, class_weight=class_weight,
						metric=metric, random_state=random_state)
		self.activation = get_activation(activation)
		self.softmax = get_activation('softmax')
		self.regularizer = get_regularizer(regularize)
		self.optimizer = get_optimizer(optimizer)
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
			self.weights_ = [self.random_state.randn(self.n_features_, self.n_classes_) * 0.1]
			self.bias_ = [self.random_state.randn(self.n_classes_) * 0.1]
			self.n_layers_ = 1
			self.layers = (self.n_features_,)
		else:
			self.weights_.append(self.random_state.randn(self.n_features_, self.layers[0]) * 0.1)
			self.bias_.append(self.random_state.randn(self.layers[0]))
			for l in range(self.n_layers_ - 1):
				self.weights_.append(self.random_state.randn(self.layers[l], self.layers[l+1]) * 0.1)
				self.bias_.append(self.random_state.randn(self.layers[l+1]))
			self.weights_.append(self.random_state.randn(self.layers[-1], self.n_classes_) * 0.1)
			self.bias_.append(self.random_state.randn(self.n_classes_) * 0.1)
			self.n_layers_ += 1
		keys = []
		for l in range(self.n_layers_):
			keys += ['w' + str(l), 'b' + str(l)]
		self.optimizer.setup(keys)

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
		for l in range(self.n_layers_):
			Z = np.dot(X, self.weights_[l]) + self.bias_[l]
			if l < self.n_layers_ - 1 : A = self.activation.activation(Z)
			else : A = self.softmax.activation(Z)
			self._z.append(Z)
			self._x.append(X)
			X = A
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
		dY_ = dY
		for l in range(self.n_layers_ - 1, -1, -1):
			if l == self.n_layers_ - 1:
				dZ = dY * self.softmax.gradient(self._z[-1])
			else : dZ = dY * self.activation.gradient(self._z[l])
			dW = np.dot(self._x[l].T, dZ) / m
			db = np.sum(dZ, axis=0) / m
			dY = np.dot(dZ, self.weights_[l].T)
			if self.regularizer is not None:
				dW += self.regularizer.gradient(self.weights_[l])
			self.weights_[l] -= self.optimizer.update('w' + str(l), dW)
			self.bias_[l] -= self.optimizer.update('b' + str(l), db)
