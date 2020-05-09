import numpy as np
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator


class Base(BaseEstimator, ABC):
	def __init__(self, verbose=0, warm_start=False, metric='accuracy',
					random_state=None):
		self.verbose = verbose
		self.warm_start = warm_start
		self.metric = get_metrics(metric)
		self.random_state = create_random_state(seed=random_state)
		self.fitted_ = False

	@abstractmethod
	def fit(self, X, Y, weights=None):
		raise NotImplementedError("No fit function implemented")

	def predict(self, X):
		pred = self.predict_proba(X)
		pred = np.argmax(pred, axis=1)
		return pred

	def predict_log_proba(self, X):
		return np.log(self.predict_proba(X))

	@abstractmethod
	def predict_proba(self, X):
		raise NotImplementedError("No predict_proba function implemented")

	def score(self, X, Y, weights=None):
		pred = self.predict(X)
		if weights is None : weights = np.ones(len(X))
		return self.metric.score(pred, Y, weights=weights)

	def _is_fitted(self):
		return self.fitted_

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
