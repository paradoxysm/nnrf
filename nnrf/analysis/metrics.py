from abc import ABC, abstractmethod
from sklearn import metrics
import numpy as np

from nnrf.utils import one_hot, check_XY
from nnrf.utils._base import Base

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
	elif isinstance(name, (type(None), Metric)) : return name
	else : raise ValueError("Invalid metric function")


class Metric(Base, ABC):
	"""
	Base Metric class.
	"""
	def __init__(self):
		super().__init__()
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
		super().__init__()
		self.name = "accuracy"

	def calculate(self, Y_hat, Y, weights=None):
		Y_hat, Y = check_XY(X=Y_hat, Y=Y)
		Y_hat, Y = np.argmax(Y_hat, axis=1), np.argmax(Y, axis=1)
		return metrics.accuracy_score(Y, Y_hat, sample_weight=weights)

class Precision(Metric):
	"""
	Precision Metric.
	"""
	def __init__(self):
		super().__init__()
		self.name = "precision"

	def calculate(self, Y_hat, Y, average='micro', weights=None):
		Y_hat, Y = check_XY(X=Y_hat, Y=Y)
		Y_hat, Y = np.argmax(Y_hat, axis=1), np.argmax(Y, axis=1)
		return metrics.precision_score(Y, Y_hat, average=average,
										sample_weight=weights)

class Recall(Metric):
	"""
	Recall Metric.
	"""
	def __init__(self):
		super().__init__()
		self.name = "recall"

	def calculate(self, Y_hat, Y, average='micro', weights=None):
		Y_hat, Y = check_XY(X=Y_hat, Y=Y)
		Y_hat, Y = np.argmax(Y_hat, axis=1), np.argmax(Y, axis=1)
		return metrics.recall_score(Y, Y_hat, average=average,
										sample_weight=weights)

class FScore(Metric):
	"""
	F-Score Metric.

	Parameters
	----------
	beta : float, default=1
		Weight of recall in F-score.
	"""
	def __init__(self, beta=1):
		super().__init__()
		self.name = "f-score"
		self.beta = beta

	def calculate(self, Y_hat, Y, average='micro', weights=None):
		Y_hat, Y = check_XY(X=Y_hat, Y=Y)
		Y_hat, Y = np.argmax(Y_hat, axis=1), np.argmax(Y, axis=1)
		return metrics.fbeta_score(Y, Y_hat, self.beta, average=average,
									sample_weight=weights)

class ROCAUC(Metric):
	"""
	Area under the Receiver Operative Curve (ROC AUC) Metric.
	"""
	def __init__(self, multi_class='ovr'):
		super().__init__()
		self.name = 'roc-auc'
		self.multi_class = multi_class

	def calculate(self, Y_hat, Y, average='macro', weights=None):
		Y_hat, Y = check_XY(X=Y_hat, Y=Y)
		Y_hat, Y = np.argmax(Y_hat, axis=1), np.argmax(Y, axis=1)
		return metrics.roc_auc_score(Y, Y_hat, average=average,
									multi_class=self.multi_class,
									sample_weight=weights)
