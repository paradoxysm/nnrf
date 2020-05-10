import numpy as np
from abc import ABC, abstractmethod

def vars_recurse(obj):
	"""
	Recursively collect vars() of the object.

	Parameters
	----------
	obj : object
		Object to collect attributes

	Returns
	-------
	params : dict
		Dictionary of object parameters.
	"""
	if hasattr(obj, '__dict__'):
		params = vars(obj)
		for k in params.keys():
			if hasattr(params[k], '__dict__'):
				params[k] = vars_recurse(params[k])
		params = dict(list(params.items()) + [('type', type(obj))])
		return params
	raise ValueError("obj does not have __dict__ attribute")

class Base:
	"""
	Base object class for nnrf.
	"""
	def __init__(self):
		pass

	def get_params(self):
		"""
		Get all parameters of the object, recursively.

		Returns
		-------
		params : dict
			Dictionary of object parameters.
		"""
		params = vars(self)
		for k in params.keys():
			if isinstance(params[k], Base):
				params[k] = k.get_params()
			elif isinstance(params[k], np.random.RandomState):
				params[k] = {'type': np.random.RandomState,
								'seed': params[k].get_state()}
			elif hasattr(params[k], '__dict__'):
				params[k] = vars_recurse(params[k])
		params = dict(list(params.items()) + [('type', type(self))])
		return params

	def set_params(self, params):
		"""
		Set the attributes of the object with the given
		parameters.

		Parameters
		----------
		params : dict
			Dictionary of object parameters.

		Returns
		-------
		self : Base
			Itself, with parameters set.
		"""
		valid = self.get_params().keys()
		for k, v in params.items():
			if k not in valid:
				raise ValueError("Invalid parameter %s for object %s" % \
									(k, self.__name__))
			param = v
			if isinstance(v, dict) and 'type' in v.keys():
				t = v['type']
				if t == np.random.RandomState:
					state = v['seed']
					param = np.random.RandomState().set_state(state)
				elif 'get_params' in dir(t) and issubclass(t, Base):
					param = t().set_params(v.pop('type'))
				else:
					param = t()
					for p, p_v in v.pop('type').items():
						setattr(param, p, p_v)
			setattr(self, k, param)
		return self


class BaseEstimator(Base, ABC):
	"""
	Base Estimator Class for nnrf.
	Implements common methods for estimators.

	Parameters
	----------
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
	"""
	def __init__(self, batch_size=None, verbose=0, warm_start=False,
					class_weight=None, metric='accuracy', random_state=None):
		self.batch_size = batch_size
		self.verbose = verbose
		self.warm_start = warm_start
		self.metric = get_metrics(metric)
		self.class_weight = class_weight
		self.random_state = create_random_state(seed=random_state)
		self.fitted_ = False

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

	def predict_log_proba(self, X):
		"""
		Predict class log-probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples,)
			Class log-probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		return np.log(self.predict_proba(X))

	@abstractmethod
	def predict_proba(self, X):
		"""
		Predict class probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples,)
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

	def _calculate_batch(self, length):
		"""
		Calculate the batch size for the data of given length.

		Parameters
		----------
		length : int
			Length of the data to be batched.

		Returns
		-------
		batch_size : int
			Batch size.
		"""
		if self.batch_size is None : return length
		elif isinstance(self.batch_size, int) and self.batch_size > 0 and \
				self.batch_size <= length:
			return self.batch_size
		elif isinstance(self.batch_size, float) and 0 < self.batch_size <= 1:
			return int(self.batch_size * length)
		else:
			raise ValueError("Batch size must be None, an int less than %d," % length,
								"or a float within (0,1]")

	def _calculate_weight(self, Y, weights=None):
		"""
		Calculate the weights applied to the predicted labels,
		combining class weights and sample weights.

		Parameters
		----------
		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		weights : array-like, shape=(n_samples,)
			Weights combining sample weights and class weights.
		"""
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
