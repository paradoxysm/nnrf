import numpy as np
from sklearn.base import BaseEstimator

from nnrf import NNDT
from nnrf.ml import get_loss, get_activation, get_regularizer
from nnrf.ml.activation import PReLU
from nnrf.utils import check_XY, one_hot, create_random_state, BatchDataset
from nnrf.analysis import get_metrics


class NNRF(BaseEstimator):
	def __init__(self, n=50, d=5, r='sqrt', rate=0.001, loss='cross-entropy',
					activation=PReLU(0.2), regularize=None, max_iter=100, tol=1e-4,
					bootstrap_size=None, batch_size=None, class_weight=None, softmax=False,
					verbose=0, warm_start=False, metric='accuracy',
					random_state=None):
		self.n = n
		self.d = d
		self.r = r
		self.rate = rate
		self.activation = get_activation(activation)
		self.loss = get_loss(loss)
		self.regularizer = get_regularizer(regularize)
		self.max_iter = max_iter
		self.tol = tol
		self.bootstrap_size = bootstrap_size
		self.batch_size = batch_size
		self.class_weight = class_weight
		self.softmax = softmax
		self.verbose = verbose
		self.warm_start = warm_start
		self.metric = get_metrics(metric)
		self.random_state = create_random_state(seed=random_state)
		self.fitted_ = False
		self.estimators_ = []
		self.weights_ = np.array([])
		self.bias_ = np.array([])
		self.n_classes_ = None
		self.n_features_ = None
		self.memory = {}

	def _initialize(self):
		random_state = self.random_state.choice(self.n*1000, size=self.n, replace=False)
		for n in range(self.n):
			if self.verbose == 0 : verbose = 0
			else : verbose = self.verbose - 1
			nndt = NNDT(d=self.d, r=self.r, rate=self.rate, loss=self.loss,
							activation=self.activation,regularize=self.regularizer,
							 max_iter=self.max_iter, tol=self.tol,
							batch_size=self.batch_size, class_weight=self.class_weight,
							verbose=verbose, warm_start=self.warm_start, metric=self.metric,
							random_state=random_state[n])
			nndt.n_classes_ = self.n_classes_
			nndt.n_features_ = self.n_features_
			self.estimators.append(nndt)
		self.weights_ = self.random_state.randn((self.n*self.n_classes, self.n_classes)) * 0.1
		self.bias_ = self.random_state.randn(self.n_classes) * 0.1

	def decision_path(self, X, full=False):
		paths = []
		for e in self.estimators_:
			paths.append(e.decision_path(X))
		paths = np.array(paths)
		if not full:
			paths = np.mean(paths, axis=0)
		return paths

	def fit(self, X, Y, weights=None):
		X, Y = check_XY(X=X, Y=Y)
		self.n_classes_ = len(set(Y))
		self.n_features_ = X.shape[1]
		try : Y = one_hot(Y, cols=self.n_classes_)
		except : raise
		if self.bootstrap_size is None:
			bootstrap = X.shape[0]
		elif isinstance(self.bootstrap_size, int) and self.bootstrap > 0:
			bootstrap = self.bootstrap_size
		elif isinstance(self.bootstrap_size, float) and 0 < self.bootstrap_size <= 1:
			bootstrap = int(self.bootstrap_size * X.shape[0])
		else : raise ValueError("Bootstrap Size must be None, a positive int or float in (0,1]")
		if self.batch_size is None : self.batch_size = len(Y)
		if not self.warm_start or not self._is_fitted():
			if verbose > 0 : print("Initializing NNRF")
			self._initialize()
		if verbose > 0 : print("Training model with %d estimators." % self.n)
		estimators = range(len(self.estimators_))
		if verbose == 1 : estimators = trange(len(self.estimators_))
		for e in estimators:
			if verbose == 1 : estimators.set_description("Estimator %d" % e+1)
			elif verbose > 1 : print("Fitting estimator %d" % e+1)
			data = np.concatenate((X, Y, weights.reshape(-1,1)), axis=1)
			data = self.random_state.permutation(data)
			X_ = data[:bootstrap, :X.shape[1]]
			Y_ = data[:bootstrap, X.shape[1]:-1]
			weights = data[:bootstrap, -1]
			e.fit(X_, Y_, weights=weights)
		if self.softmax : self._fit_softmax(X, Y, weights=weights)
		self.fitted_ = True
		if verbose > 0 : print("Training complete.")
		return self

	def predict(self, X):
		pred = self.predict_proba(X)
		pred = np.argmax(pred, axis=1)
		return pred

	def predict_log_proba(self, X):
		return np.log(self.predict_proba(X))

	def predict_proba(self, X):
		if not self._is_fitted():
			raise RunTimeError("Model is not fitted")
		X = check_XY(X=X)
		if verbose > 0 : print("Predicting %d samples." % \
								X.shape[0])
		if self.softmax:
			pred = self._predict_base(X)
			pred, _ = self._predict_softmax(pred)
		else:
			pred = np.zeros((len(X), self.n_classes_))
			for e in self.estimators_:
				pred += e.predict_proba(X)
				pred /= self.n
		return pred

	def score(self, X, Y, weights=None):
		pred = self.predict(X)
		if weights is None : weights = np.ones(len(X))
		return self.metric.score(pred, Y, weights=weights)

	def set_warm_start(self, warm):
		for e in self.estimators_:
			e.warm_start = warm

	def _is_fitted(self):
		estimators = len(self.estimators_) > 0
		return estimators and self.fitted_

	def _fit_softmax(X, Y, weights=None):
		if self.batch_size is None : self.batch_size = len(Y)
		X = self._predict_base(X)
		ds = BatchDataset(X, Y, seed=self.random_state).shuffle().repeat().batch(self.batch_size)
		if verbose > 0 : print("Training softmax aggregation layer")
		loss_prev, early_stop, e = np.inf, False, 0
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
				Y_hat, Z = self._predict_softmax(X_batch)
				loss = np.mean(self.loss.loss(Y_hat, Y_batch))
				metric = self.metric.calculate(Y_hat, Y_batch)
				msg = 'loss: %.4f' % loss + ', ' + self.metric.name + ': %.4f' % metric
				if verbose == 1 : epochs.set_description(msg)
				if verbose == 2 : batches.set_description(msg)
				elif verbose > 2 : print("Epoch %d, Batch %d completed." % (e, b), msg)
				if tol is not None and np.abs(loss - loss_prev) < tol:
					early_stop = True
					break
				self._backward_softmax(Y_hat, Y_batch, Z, X_batch, weights=weights)
				loss_prev = loss
			if early_stop : break
		return self

	def _backward_softmax(Y_hat, Y, Z, X, weights=None):
		weights = self._calculate_weight(Y, weights=weights)
		m = len(X)
		dY = self.loss.gradient(Y_hat, Y) * weights
		softmax = get_activation('softmax')
		dZ = dY * softmax.gradient(Z)
		dW = np.dot(X.T, dZ) / m
		db = np.sum(dZ, axis=0) / m
		if self.regularizer is not None:
			self.weights_ -= self.regularizer.gradient(self.weights_)
		self.weights_ -= self.rate * dW
		self.bias_ -= self.rate * db

	def _predict_base(X):
		pred = np.array([])
		for e in self.estimators_:
			pred = np.concatenate((pred, e.predict_proba(X)), axis=1)
		return pred

	def _predict_softmax(X):
		Z = np.dot(self.weights_, pred) + self.bias_
		A = get_activation('softmax').activation(Z)
		return A, Z

	def _calculate_weight(self, Y, weights=None):
		if weights is None : weights = np.ones(len(Y))
		d = self.class_weight
		if isinstance(d, str) and d == 'balanced':
			l = len(Y) / (self.n_classes * np.bincount(Y))
			d = {k: l[k] for k in range(len(l))}
		elif isinstance(d, dict):
			k = list(d.keys())
			class_weights = np.where(Y == k, self.class_weight[k])
		elif d is None : class_weights = np.ones(len(Y))
		else : raise ValueError("Class Weight must either be a dict or 'balanced' or None")
		return weights * class_weights
