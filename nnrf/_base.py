import numpy as np
from abc import ABC, abstractmethod

class Base:
	def __init__(self):
		pass

	def get_params(self):
		params = vars(self)
		for k in params.keys():
			if isinstance(params[k], Base):
				params[k] = k.get_params()
			elif isinstance(params[k], np.random.RandomState):
				params[k] = {'type': np.random.RandomState,
								'seed': params[k].get_state()}
			elif hasattr(params[k], '__dict__'):
				params[k] = dict(list(vars(params[k]).keys()) + \
								[('type', type(params[k]))])
		params = dict(list(params.items()) + [('type', type(self))])
		return params

	def set_params(self, params):
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
			setattr(self, k, v)
		return self


class BaseEstimator(Base, ABC):
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
