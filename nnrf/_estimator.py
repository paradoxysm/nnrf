import numpy as np
from tqdm import trange
from abc import ABC, abstractmethod

from nnrf.utils import calculate_batch, create_random_state, \
						BatchDataset, one_hot, check_XY
from nnrf.analysis import get_metrics
from nnrf.ml import get_loss

from nnrf._base import Base

class BaseEstimator(Base, ABC):
	"""
	Base Estimator Class.
	Implements common methods for estimators.

	Parameters
	----------
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
	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.

	n_classes_ : int
		Number of classes.

	n_features : int
		Number of features accepted as input.
	"""
	def __init__(self, verbose=0, warm_start=False, metric='accuracy'):
		self.verbose = verbose
		self.warm_start = warm_start
		self.metric = get_metrics(metric)
		self.fitted_ = False
		self.n_classes_ = None
		self.n_features_ = None

	@abstractmethod
	def fit(self, X, Y, weights=None):
		"""
		Train the model on the given data and labels.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Training data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		self : Base
			Fitted estimator.
		"""
		raise NotImplementedError("No fit function implemented")

	def predict(self, X):
		"""
		Predict classes for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y : array-like, shape=(n_samples,)
			Predicted labels.
		"""
		pred = self.predict_proba(X)
		pred = np.argmax(pred, axis=1)
		return pred

	def predict_log_proba(self, X, *args, **kwargs):
		"""
		Predict class log-probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples, n_classes)
			Class log-probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		return np.log(self.predict_proba(X))

	@abstractmethod
	def predict_proba(self, X, *args, **kwargs):
		"""
		Predict class probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples, n_classes)
			Class probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		raise NotImplementedError("No predict_proba function implemented")

	def score(self, Y, X=None, Y_hat=None, weights=None):
		"""
		Return mean metric of the estimator on the given
		data/predictions and target labels.

		If both data and predictions are provided, `score`
		just uses the predictions.

		Parameters
		----------
		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		X : array-like, shape=(n_samples, n_features), default=None
			Data to predict.

		Y_hat : array-like, shape=(n_samples,), default=None
			Predicted labels.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		score : float
			Mean metric score of the estimator for the given
			data/labels.
		"""
		if X is None and Y_hat is None:
			raise ValueError("Either X or Y_hat must be provided")
		elif Y_hat is None:
			Y_hat = self.predict(X)
		if self.metric is not None:
			if weights is None : weights = np.ones(len(Y_hat))
			return self.metric.score(Y_hat, Y, weights=weights)
		return 0

	def _is_fitted(self):
		"""
		Return True if the model is properly ready
		for prediction.

		Returns
		-------
		fitted : bool
			True if the model can be used to predict data.
		"""
		return self.fitted_


class BaseClassifier(BaseEstimator):
	"""
	Base Estimator Class.
	Implements common methods for estimators.

	Parameters
	----------
	loss : str, LossFunction, default='cross-entropy'
		Loss function to use for training. Must be
		one of the default loss functions or an object
		that extends LossFunction.

	max_iter : int, default=10
		Maximum number of epochs to conduct during training.

	tol : float, default=1e-4
		Convergence criteria for early stopping.

	batch_size : int, float, default=None
		Batch size for training. Must be one of:
		 - int : Use `batch_size`.
		 - float : Use `batch_size * n_samples`.
		 - None : Use `n_samples`.

	verbose : int, default=0
		Verbosity of estimator; higher values result in
		more verbose output.

	warm_start : bool, default=False
		Determines warm starting to allow training to pick
		up from previous training sessions.

	class_weight : dict, 'balanced', or None, default=None
		Weights associated with classes in the form
		`{class_label: weight}`. Must be one of:
		 - None : All classes have a weight of one.
		 - 'balanced': Class weights are automatically calculated as
						`n_samples / (n_samples * np.bincount(Y))`.

	metric : str, Metric, or None, default='accuracy'
		Metric for estimator score.

	random_state : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	Attributes
	----------
	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.

	n_classes_ : int
		Number of classes.

	n_features : int
		Number of features accepted as input.
	"""
	def __init__(self, loss='cross-entropy', max_iter=100, tol=1e-4,
					batch_size=None, verbose=0, warm_start=False,
					class_weight=None, metric='accuracy', random_state=None):
		super().__init__(verbose=0, warm_start=False, metric='accuracy')
		self.loss = get_loss(loss)
		self.max_iter = max_iter
		self.tol = tol
		self.batch_size = batch_size
		self.class_weight = class_weight
		self.random_state = create_random_state(seed=random_state)
		self._x = []
		self._z = []

	@abstractmethod
	def _initialize(self):
		"""
		Initialize the parameters of the NNRF.
		"""
		raise NotImplementedError("No initialize function implemented")

	def fit(self, X, Y, weights=None):
		"""
		Train the model on the given data and labels.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Training data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		self : Base
			Fitted estimator.
		"""
		X, Y = check_XY(X=X, Y=Y)
		if self.n_classes_ is None : self.n_classes_ = len(set(Y))
		if self.n_features_ is None : self.n_features_ = X.shape[1]
		try : Y = one_hot(Y, cols=self.n_classes_)
		except : raise
		batch_size = calculate_batch(self.batch_size, len(Y))
		ds = BatchDataset(X, Y, seed=self.random_state).shuffle().repeat().batch(self.batch_size)
		if not self.warm_start or not self._is_fitted():
			if verbose > 0 : print("Initializing model")
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
				metric = self.score(Y_batch, Y_hat=Y_hat, weights=weights)
				msg = 'loss: %.4f' % loss + ', ' + self.metric.name + ': %.4f' % metric
				if verbose == 1 : epochs.set_description(msg)
				elif verbose == 2 : batches.set_description(msg)
				elif verbose > 2 : print("Epoch %d, Batch %d completed." % (e, b), msg)
				if self.tol is not None and np.abs(loss - loss_prev) < self.tol:
					early_stop = True
					break
				self._backward(Y_hat, Y_batch, weights=weights)
				loss_prev = loss
			if early_stop : break
		self.fitted_ = True
		if verbose > 0 : print("Training complete.")
		return self

	def predict_proba(self, X, *args, **kwargs):
		"""
		Predict class probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples, n_classes)
			Class probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		if not self._is_fitted():
			raise RunTimeError("Model is not fitted")
		X = check_XY(X=X)
		if verbose > 0 : print("Predicting %d samples." % \
								X.shape[0])
		return self._forward(X)

	def score(self, Y, X=None, Y_hat=None, weights=None):
		"""
		Return mean metric of the estimator on the given
		data/predictions and target labels.

		If both data and predictions are provided, `score`
		just uses the predictions.

		Parameters
		----------
		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		X : array-like, shape=(n_samples, n_features), default=None
			Data to predict.

		Y_hat : array-like, shape=(n_samples,), default=None
			Predicted labels.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		score : float
			Mean metric score of the estimator for the given
			data/labels.
		"""
		if X is None and Y_hat is None:
			raise ValueError("Either X or Y_hat must be provided")
		elif Y_hat is None:
			Y_hat = self.predict(X)
		if self.metric is not None:
			if weights is None : weights = np.ones(len(Y_hat))
			return self.metric.score(Y_hat, Y, weights=weights)
		return 0

	@abstractmethod
	def _forward(self, X, *args, **kwargs):
		"""
		Conduct the forward propagation steps through the model.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Output.
		"""
		raise NotImplementedError("No forward propagation implemented")

	def _backward(self, Y_hat, Y, *args, weights=None, **kwargs):
		"""
		Conduct the backward propagation steps through the
		model.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Model output.

		Y : array-like, shape=(n_samples, n_classes)
			Target labels in one hot encoding.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.
		"""
		return
