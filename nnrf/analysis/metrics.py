import numpy as np
from abc import ABC, abstractmethod

from nnrf._base import Base

def get_metrics(name):
	"""
	Lookup table of default metrics.

	Parameters
	----------
	name : Metric, None, str
		Metric to look up. Must be one of:
		 - 'accuracy' : Accuracy.
		 - 'precision' : Precision.
		 - 'recall' : Recall.
		 - 'f-score' : F1-Score.
		 - 'roc-auc' : ROC-AUC.
		 - Metric : A custom implementation.
		 - None : Return None.
		Custom Metrics must implement `score` which
		by default should return a single float value.

	Returns
	-------
	metric : Metric or None
		The metric.
	"""
	if name == 'accuracy' : return Accuracy()
	elif name == 'precision' : return Precision()
	elif name == 'recall' : return Recall()
	elif name == 'f-score' : return FScore()
	elif name == 'roc-auc' : return ROCAUC()
	elif isinstance(name, (None, Metric)) : return name
	else : raise ValueError("Invalid metric function")


class Metric(Base, ABC):
	"""
	Base Metric class.
	"""
	def __init__(self):
		self.name = "metric"

	@abstractmethod
	def calculate(self, Y_hat, Y, average='micro', weights=None):
		"""
		Calculate metric of given labels, `Y_hat`,
		compared to ground truth, `Y`. By default, gives
		overall metric, or micro-average.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples,)
			Prediction labels.

		Y : array-like, shape=(n_samples,)
			Ground truth labels.

		average : {'micro', 'macro', 'weighted', None}, default='micro'
			Method to average metric. Must be one of:
			 - 'micro' : Overall metric.
			 - 'macro' : Unweighted mean metric across classes.
			 - 'weighted' : Weighted mean metric across classes.
			 - None : Metrics for each class.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights.

		Returns
		-------
		metric : dict, ndarray, float
			The metric conducted with the given method.
			Returns a dictionary if `average` is None,
			an ndarray if `average` is `macro` or `weighted`,
			or a float if `average` is `micro` or if there are
			no samples.
		"""
		raise NotImplementedError("No calculate function implemented")

	def score(self, Y_hat, Y, weights=None):
		"""
		Calculate overall score of given labels, `Y_hat`,
		compared to ground truth, `Y`.

		Parameters
		----------
		Y_hat : array-like, shape=(n_samples,)
			Prediction labels.

		Y : array-like, shape=(n_samples,)
			Ground truth labels.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights.

		Returns
		-------
		score : float, range=[0,1]
			The score.
		"""
		return self.calculate(Y_hat, Y, weights=weights)

class Accuracy(Metric):
	"""
	Accuracy Metric.
	"""
	def __init__(self):
		self.name = "accuracy"

	def calculate(self, Y_hat, Y, average='micro', weights=None):
		Y_hat, Y = check_XY(X=Y_hat, Y=Y)
		if weights is None : weights = np.ones(len(Y_hat)) / len(Y_hat)
		elif weights.shape != Y_hat.shape:
			raise ValueError("Weights must be the same shape as labels.",
								"Expected", str(Y_hat.shape), " but got",
								weights.shape)
		else : weights /= np.sum(weights)
		if average is 'micro':
			return (Y_hat == Y).all(axis=0).mean()
		elif average is None or average == 'macro' or average == 'weighted':
			classes = sorted(set(Y))
			accuracies = {c: 0 for c in classes}
			supports = {c: 0 for c in classes}
			for l, t, w in zip(Y_hat_, Y, weights):
				supports[t] += 1
				if l == t : accuracies[t] += w
			if average is None : return accuracies
			accuracies = [accuracies[c] for c in classes]
			weights = None
			if average == 'weighted':
				weights = [supports[c] for c in classes]
				if np.sum(weights) == 0:
					return 0
			return np.average(accuracies, weights=weights)

class Precision(Metric):
	"""
	Precision Metric.
	"""
	def __init__(self):
		self.name = "precision"

	def calculate(self, Y_hat_, Y, average='micro', weights=None):
		pass


class Recall(Metric):
	"""
	Recall Metric.
	"""
	def __init__(self):
		self.name = "recall"

	def calculate(self, Y_hat_, Y, average='micro', weights=None):
		pass


class FScore(Metric):
	"""
	F-Score Metric.

	Parameters
	----------
	beta : float, default=1
		Weight of recall in F-score.
	"""
	def __init__(self, beta=1):
		self.name = "f-score"
		self.beta = beta

	def calculate(self, Y_hat_, Y, average='micro', weights=None):
		pass


class ROCAUC(Metric):
	"""
	Area under the Receiver Operative Curve (ROC AUC) Metric.
	"""
	def __init__(self):
		self.name = 'roc-auc'

	def calculate(self, Y_hat_, Y, average='micro', weights=None):
		pass
