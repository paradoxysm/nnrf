import numpy as np

from nnrf.ml import get_loss, get_activation, get_regularizer
from nnrf.ml.activation import PReLU
from nnrf.utils import check_XY, one_hot, create_random_state, BatchDataset
from nnrf.analysis import get_metrics

from nnrf._base import BaseEstimator


class NNDT(BaseEstimator):
	def __init__(self, d=5, r='sqrt', loss='cross-entropy',
					activation=PReLU(0.2), regularize=None, rate=0.001,
					max_iter=100, tol=1e-4, batch_size=None, class_weight=None,
					verbose=0, warm_start=False, metric='accuracy',
					random_state=None):
		super().__init__(verbose=verbose, warm_start=warm_start, metric=metric,
						random_state=random_state)
		self.d = d
		self.r = r
		self.activation = get_activation(activation)
		self.softmax = get_activation('softmax')
		self.loss = get_loss(loss)
		self.regularizer = get_regularizer(regularize)
		self.rate = rate
		self.max_iter = max_iter
		self.tol = tol
		self.batch_size = batch_size
		self.class_weight = class_weight
		self.weights_ = np.array([])
		self.bias_ = np.array([])
		self.inputs_ = np.array([])
		self.sweights_ = np.array([])
		self.sbias_ = np.array([])
		self.n_classes_ = None
		self.n_features_ = None
		self._p = []
		self._s = []
		self._z = []
		self._x = []
		self._memory = {}

	def _initialize(self):
		self._calculate_r()
		n, s = 2**self.d - 1, self.r + 1
		self.weights_ = self.random_state.randn((n, s, 2)) * 0.1
		self.bias_ = self.random_state.randn((n, 1, 2)) * 0.1
		for i in range(n):
			input = self.random_state.choice(self.n_features, size=(1, self.r), replace=False)
			self.inputs_ = np.concatenate((self.inputs_, input))
		self.sweights_ = self.random_state.randn((2**self.d, self.n_classes_)) * 0.1
		self.sbias_ = self.random_state.randn(self.n_classes_) * 0.1

	def decision_path(self, X):
		pred = self._forward(X)
		paths = np.array([np.zeros(len(X))] + self._s[:-1])[:,np.newaxis]
		return np.concatenate(paths).T

	def fit(self, X, Y, weights=None):
		X, Y = check_XY(X=X, Y=Y)
		if self.n_classes_ is None : self.n_classes_ = len(set(Y))
		if self.n_features_ is None : self.n_features_ = X.shape[1]
		try : Y = one_hot(Y, cols=self.n_classes_)
		except : raise
		if self.batch_size is None : self.batch_size = len(Y)
		ds = BatchDataset(X, Y, seed=self.random_state).shuffle().repeat().batch(self.batch_size)
		if not self.warm_start or not self._is_fitted():
			if verbose > 0 : print("Initializing NNDT")
			self._initialize()
		if verbose > 0 : print("Training model for %d epochs" % self.max_iter,
								"on %d samples in batches of %d." % \
								(X.shape[0], self.batch_size),
								"Convergence tolerance set to %.4f." % self.tol)
		loss_prev, early_stop, e = np.inf, False, 0
		epochs = range(self.max_iter)
		if verbose == 1 : epochs = trange(self.max_iter)
		for e in epochs:
			batches = range(ds.n_batches)
			if verbose == 2 : batches = trange(batches)
			if verbose > 2 : print("Epoch %d" % e)
			for b in batches:
				X_batch, Y_batch = ds.next()
				if X_batch == []:
					if verbose > 0 : print("No more data to train. Ending training.")
					early_stop = True
					break
				Y_hat = self._forward(X_batch)
				loss = np.mean(self.loss.loss(Y_hat, Y_batch))
				metric = self.metric.calculate(Y_hat, Y_batch)
				msg = 'loss: %.4f' % loss + ', ' + self.metric.name + ': %.4f' % metric
				if verbose == 1 : epochs.set_description(msg)
				elif verbose == 2 : batches.set_description(msg)
				elif verbose > 2 : print("Epoch %d, Batch %d completed." % (e, b), msg)
				if tol is not None and np.abs(loss - loss_prev) < tol:
					early_stop = True
					break
				self._backward(Y_hat, Y_batch, weights=weights)
				loss_prev = loss
			if early_stop : break
		self.fitted_ = True
		if verbose > 0 : print("Training complete.")
		return self

	def predict_proba(self, X):
		if not self._is_fitted():
			raise RunTimeError("Model is not fitted")
		X = check_XY(X=X)
		if verbose > 0 : print("Predicting %d samples." % \
								X.shape[0])
		return self._forward(X)

	def _is_fitted(self):
		weights = len(self.weights_) > 0
		bias = len(self.bias_) > 0
		inputs = len(self.inputs_) > 0
		return weights and bias and inputs and self.fitted_

	def _forward(self, X):
		s = np.zeros(len(X))
		X_ = np.concatenate((self._get_inputs(0, X), np.zeros(len(X))), axis=1)
		self._x, self._z, self._s, self._p = [], [], [], []
		for d in range(self.d):
			n = 2**d + s - 1
			weights = self._get_weights(n)
			bias = self._get_bias(n)
			X_d = self._get_inputs(n, X_)
			z = np.einsum('ijk,ij->ik', weights, X_d) + bias
			a = self.activation.activation(z)
			k = np.argmax(a, axis=0)
			s = 2 * s + k
			i = (np.arange(len(X)), k)
			p = a[i].reshape(-1,1)
			X_ = np.concatenate((X, p), axis=1)
			self._x.append(X_d)
			self._z.append(z)
			self._s.append(s)
			self._p.append(p)
		p = np.zeros((len(X), 2**self.d))	# N(2**d)
		s = 2 * self._s[-2]
		s = np.concatenate((s, s+1))
		i = (np.repeat(np.arange(len(X)),2), s)
		p[i] += a
		z = np.dot(p, self.sweights_) + self.sbias_ # NC
		Y_hat = self.softmax.activation(z)
		self._z.append(z)
		self._p.append(p)
		return Y_hat

	def _backward(self, Y_hat, Y, weights=None):
		weights = self._calculate_weight(Y, weights=weights)
		dY = self.loss.gradient(Y_hat, Y) * weights
		dsZ = dY * self.softmax.gradient(self._z[-1]) # NC
		dsW = np.dot(self._p[-1].T, dsZ) / len(Y_hat) # (2**d)C
		dsb = np.sum(dsZ, axis=0) / len(Y_hat) # C
		dp_ = np.dot(dsZ, self.sweights_.T) # N(2**d)
		s = 2 * self._s[-2]
		s = np.concatenate((s, s+1))
		i = (np.repeat(np.arange(len(X)),2), s)
		dp = dp_[i].reshape(-1,2) # N2
		if self.regularizer is not None:
			self.sweights_ -= self.regularizer.gradient(self.sweights_)
			for n in range(len(self.weights_)):
				self.weights_[n] -= self.regularizer.gradient(self.weights_[n])
		self.sweights_ -= self.rate * dsW
		self.sbias_ -= self.rate * dsb
		for i in range(self.d):
			z = self._z[-2-i]
			if i > 0:
				idx = tuple([np.arange(len(z)), self._s[-1-i] % 2])
				dZ_ = np.sum(dp, axis=1) * self.activation.gradient(z[idx]) # N1 * N1 -> N1
				dZ = np.zeros(z.shape)
				dZ[idx] += dZ_.reshape(-1) # N2
			else:
				dZ = dp * self.activation.gradient(z) # N2 * N2 -> N2
			dW = np.einsum('ij,ik->ijk', self._x[-1-i], dZ) # NM * N2 -> NM2
			db = dZ # N2
			n = 2**d + self._s[-2-i] - 1
			dp = np.einsum('ijk,ik->ik', self._get_weights(n), dZ) # NM2 * N2 -> N2
			n_count =  np.bincount(n.reshape(-1))
			self.weights_[n] -= self.rate * dW / n_count[n]
			self.bias_[n] -= self.rate * db / n_count[n]

	def _get_weights(self, node):
		return self.weights_[node]

	def _get_bias(self, node):
		return self.bias_[node]

	def _get_inputs(self, node, X):
		return X[i, self.inputs_[node]]

	def _calculate_r(self):
		if self.n_features_ is None:
			raise ValueError("Number of features needs to be set first to calculate r")
		if self.r == 'sqrt' : self.r = int(np.sqrt(self.n_features_))
		elif self.r == 'log2' : self.r = int(np.log(self.n_features_) / np.log(2))
		elif isinstance(self.r, float) and 0 < self.r <= 1 : self.r = int(self.r * self.n_features_)
		elif not (isinstance(self.r, int) and self.r > 0):
			raise ValueError("R must be None, a positive int or float in (0,1]")
