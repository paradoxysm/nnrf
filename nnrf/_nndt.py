import numpy as np

from nnrf.ml import get_activation, get_regularizer, get_optimizer
from nnrf.ml.activation import PReLU
from nnrf.utils import check_XY, one_hot, decode, calculate_weight, \
						create_random_state, BatchDataset
from nnrf.analysis import get_metrics

from nnrf.utils._estimator import BaseClassifier


class NNDT(BaseClassifier):
	"""
	Neural Network structured as a Decision Tree.

	Parameters
	----------
	d : int, default=5
		Depth, or number of layers, of the NNDT.

	r : int, float, None, or {'sqrt', 'log2'}, default='sqrt'
		The number of features provided as input to each
		node. Must be one of:
		 - int : Use `r` features.
		 - float : Use `r * n_features` features.
		 - 'sqrt' : Use `sqrt(n_features)` features.
		 - 'log2' : Use `log2(n_features)` features.
		 - None : Use all features.

	loss : str, LossFunction, default='cross-entropy'
		Loss function to use for training. Must be
		one of the default loss functions or an object
		that extends LossFunction.

	activation : str, Activation, default=PReLU(0.2)
		Activation function to use at each node in the NNDT.
		Must be one of the default loss functions or an
		object that extends Activation.

	optimizer : str, Optimizer, default='adam'
		Optimization method. Must be one of
		the default optimizers or an object that
		extends Optimizer.

	regularize : str, Regularizer, None, default=None
		Regularization function to use at each node in the NNDT.
		Must be one of the default regularizers or an object that
		extends Regularizer. If None, no regularization is done.

	alpha : float, default=0.001
		Learning rate. NNDT uses gradient descent.

	max_iter : int, default=10
		Maximum number of epochs to conduct during training.

	tol : float, default=1e-4
		Convergence criteria for early stopping.

	batch_size : int, float, default=None
		Batch size for training. Must be one of:
		 - int : Use `batch_size`.
		 - float : Use `batch_size * n_samples`.
		 - None : Use `n_samples`.

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

	random_state : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	Attributes
	----------
	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.

	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.

	weights_ : ndarray, shape=(2**d - 1, r + 1, 2)
		Weights of the NNDT. Note that the root input node
		contains an extra dead input to allow simpler
		weight storage.

	bias_ : ndarray, shape=(2**d - 1, 1, 2)
		Biases of the NNDT.

	sweights_ : ndarray, shape=(2**(d-1), n_classes)
		Weights of the softmax aggregator layer.

	sbias_ : ndarray, shape=(n_classes,)
		Biases of the softmax aggregator layer.
	"""
	def __init__(self, d=5, r='sqrt', loss='cross-entropy',
					activation=PReLU(0.2), regularize=None, optimizer='adam',
					max_iter=100, tol=1e-4, batch_size=None, class_weight=None,
					verbose=0, warm_start=False, metric='accuracy',
					random_state=None):
		super().__init__(loss=loss, max_iter=max_iter, tol=tol,
						batch_size=batch_size, verbose=verbose,
						warm_start=warm_start, class_weight=class_weight,
						metric=metric, random_state=random_state)
		self.d = d
		self.r = r
		self.activation = get_activation(activation)
		self.softmax = get_activation('softmax')
		self.regularizer = get_regularizer(regularize)
		self.optimizer = get_optimizer(optimizer)
		self.weights_ = np.array([])
		self.bias_ = np.array([])
		self.inputs_ = np.array([])
		self.sweights_ = np.array([])
		self.sbias_ = np.array([])
		self._p = []
		self._s = []

	def _initialize(self):
		"""
		Initialize the parameters of the NNDT.
		"""
		self._calculate_r()
		n, s = 2**self.d - 1, self.r + 1
		self.weights_ = self.random_state.randn(n, s, 2) * 0.1
		self.bias_ = self.random_state.randn(n, 1, 2) * 0.1
		self.inputs_ = self.inputs_.reshape(-1, self.r).astype(int)
		for i in range(n):
			input = self.random_state.choice(self.n_features_, size=(1, self.r), replace=False)
			self.inputs_ = np.concatenate((self.inputs_, input))
		self.sweights_ = self.random_state.randn(2**self.d, self.n_classes_) * 0.1
		self.sbias_ = self.random_state.randn(self.n_classes_) * 0.1
		keys = ['sw', 'sb']
		for i in range(n):
			keys += ['w' + str(i), 'b' + str(i)]
		self.optimizer.setup(keys)

	def decision_path(self, X):
		"""
		Returns the nodes in each layer that comprises
		the decision path of the NNDT.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		paths : ndarray, shape=(n_samples, n_layers)
			List of paths for each data sample.
			Each path consists of a list of nodes active
			in each layer (e.g. if the third element is
			2, the third node in the third layer was active).
		"""
		pred = self._forward(X)
		paths = np.array([np.zeros(len(X))] + self._s[:-1])[:,np.newaxis]
		return np.concatenate(paths).T

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
		inputs = len(self.inputs_) > 0
		return weights and bias and inputs and self.fitted_

	def _forward(self, X):
		"""
		Conduct the forward propagation steps through the NNDT.
		Only one node in each layer is active and activates only
		one child node.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Softmax output.
		"""
		s = np.zeros(len(X))
		p = s.reshape(-1,1)
		self._x, self._z, self._s, self._p = [], [], [s], [p]
		for d in range(self.d):
			n = (2**d + s - 1).astype(int)
			weights = self._get_weights(n)
			bias = self._get_bias(n).reshape(-1,2)
			X_d = self._get_inputs(n, X)
			X_d = np.concatenate((X_d, p), axis=1)
			z = np.einsum('ijk,ij->ik', weights, X_d) + bias
			a = self.activation.activation(z)
			k = np.argmax(a, axis=1).astype(int)
			s = (2 * s + k).astype(int)
			i = (np.arange(len(X)), k)
			p = a[i].reshape(-1,1)
			self._x.append(X_d)
			self._z.append(z)
			self._s.append(s)
			self._p.append(p)
		p = np.zeros((len(X), 2**self.d))	# N(2**d)
		s = (2 * self._s[-2]).reshape(-1,1)
		s = np.concatenate((s, s+1), axis=1).astype(int)
		i = (np.arange(len(X)).reshape(-1,1), s)
		p[i] += a
		z = np.dot(p, self.sweights_) + self.sbias_ # NC
		Y_hat = self.softmax.activation(z)
		self._z.append(z)
		self._p.append(p)
		return Y_hat

	def _backward(self, Y_hat, Y, weights=None):
		"""
		Conduct the backward propagation steps through the NNDT.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Softmax output.

		Y : array-like, shape=(n_samples, n_classes)
			Target labels in one hot encoding.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.
		"""
		weights = calculate_weight(decode(Y), self.n_classes_,
					class_weight=self.class_weight, weights=weights)
		dY = self.loss.gradient(Y_hat, Y) * weights.reshape(-1,1)
		dsZ = dY * self.softmax.gradient(self._z[-1]) # NC
		dsW = np.dot(self._p[-1].T, dsZ) / len(Y_hat) # (2**d)C
		dsb = np.sum(dsZ, axis=0) / len(Y_hat) # C
		s = (2 * self._s[-2]).reshape(-1,1) # N1
		s = np.concatenate((s, s+1), axis=1).astype(int).flatten() # (2N)
		weights = self.sweights_[s].reshape(-1,2,self.sweights_.shape[-1]) # N2C
		dp = np.einsum('ik,ijk->ij', dsZ, weights) # NC * N2C -> N2
		if self.regularizer is not None:
			dsW += self.regularizer.gradient(self.sweights_)
		self.sweights_ -= self.optimizer.update('sw', dsW)
		self.sbias_ -= self.optimizer.update('sb', dsb)
		for i in range(self.d):
			z = self._z[-2-i] # N2
			if i > 0:
				idx = tuple([np.arange(len(z)), self._s[-1-i] % 2])
				dZ_ = np.sum(dp, axis=1) * self.activation.gradient(z[idx]) # N1 * N1 -> N1
				dZ = np.zeros(z.shape) # N2
				dZ[idx] += dZ_.reshape(-1) # N2
			else:
				dZ = dp * self.activation.gradient(z) # N2 * N2 -> N2
			dW_ = np.einsum('ij,ik->ijk', self._x[-1-i], dZ) # NM * N2 -> NM2
			db_ = dZ.reshape(-1,1,2) # N12
			nodes = self._s[-2-i].astype(int)
			n = (2**(self.d - i - 1) + nodes - 1).astype(int)
			weights = self._get_weights(n)[:,-1,:].reshape(-1,1,2) # N12
			dp = np.einsum('ik,ijk->ij', dZ, weights).reshape(-1,1) # N2 * N12 -> N1
			n_count =  np.bincount(nodes.reshape(-1))
			layer_size = 2**(self.d - i - 1)
			dW = np.zeros((layer_size, self.r + 1, 2))
			db = np.zeros((layer_size, 1, 2))
			if self.regularizer is not None:
				reg = self.regularizer.gradient(self.weights_)[n] # NM2
				reg = np.einsum('ijk,i->ijk', reg, 1 / n_count[nodes])
				np.add.at(dW, nodes, reg)
			np.add.at(dW, nodes, np.einsum('ijk,i->ijk', dW_, 1 / n_count[nodes]))
			np.add.at(db, nodes, np.einsum('ijk,i->ijk', db_, 1 / n_count[nodes]))
			for l in range(layer_size):
				node = int(2**(self.d - i - 1) + l - 1)
				l_key = str(node)
				self.weights_[node] -= self.optimizer.update('w' + l_key, dW[l])
				self.bias_[node] -= self.optimizer.update('b' + l_key, db[l])

	def _get_weights(self, node):
		"""
		Get weight matrices for each node given
		by `node`.

		Parameters
		----------
		node : array-like, shape=(n_samples,)
			List of nodes of interest.

		Returns
		-------
		weights : array-like, shape=(n_samples, r + 1, 2)
			List of weight matrices for each data sample.
		"""
		return self.weights_[node]

	def _get_bias(self, node):
		"""
		Get bias matrices for each node given
		by `node`.

		Parameters
		----------
		node : array-like, shape=(n_samples,)
			List of nodes of interest.

		Returns
		-------
		bias : array-like, shape=(n_samples, 1, 2)
			List of bias matrices for each data sample.
		"""
		return self.bias_[node]

	def _get_inputs(self, node, X):
		"""
		Get input features for each node given
		by `node`.

		Parameters
		----------
		node : array-like, shape=(n_samples,)
			List of nodes of interest.

		Returns
		-------
		inputs : array-like, shape=(n_samples, r)
			List of inputs for each data sample.
		"""
		indices = (np.arange(len(X)).reshape(-1,1), self.inputs_[node])
		return X[indices]

	def _calculate_r(self):
		"""
		Calculate the number of features provided
		to each node in the NNDT.
		"""
		if self.n_features_ is None:
			raise ValueError("Number of features needs to be set first to calculate r")
		if self.r is None : self.r = self.n_features_
		elif self.r == 'sqrt' : self.r = int(np.sqrt(self.n_features_))
		elif self.r == 'log2' : self.r = int(np.log(self.n_features_) / np.log(2))
		elif isinstance(self.r, float) and 0 < self.r <= 1 : self.r = int(self.r * self.n_features_)
		elif not (isinstance(self.r, int) and self.r > 0):
			raise ValueError("R must be None, a positive int or float in (0,1]")
		if self.r == 0 : self.r = 1
