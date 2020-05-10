import numpy as np
from abc import ABC, abstractmethod

from nnrf._base import Base

def get_metrics(name):
	if name == 'accuracy' : return Accuracy
	elif name == 'precision' : return Precision
	elif name == 'recall' : return Recall
	elif name == 'f-score' : return FScore
	elif isinstance(name, (None, Metric)) : return name
	else : raise ValueError("Invalid metric function")


class Metric(Base, ABC):
	def __init__(self):
		self.name = "metric"

	@abstractmethod
	def calculate(self, Y_hat, Y, *args, **kwargs):
		raise NotImplementedError("No calculate function implemented")

	@abstractmethod
	def score(self, Y_hat, Y, *args, **kwargs):
		raise NotImplementedError("No score function implemented")

class Accuracy(Metric):
	def __init__(self):
		self.name = "accuracy"

	def calculate(self, Y_hat, Y, average='micro'):
		Y_hat, Y = check_XY(X=Y_hat, Y=Y)
		if average is 'micro':
			return (Y_hat == Y).all(axis=0).mean()
		elif average is None or average == 'macro' or average == 'weighted':
			classes = sorted(set(Y))
			correct = {c: [] for c in classes}
			supports = {c: 0 for c in classes}
			for l, t in zip(Y_hat_, Y):
				supports[t] += 1
				if l == t : correct[t] += 1
			accuracies = [correct[c] / supports[c] for c in classes]
			if average is None : return accuracies
			weights = None
			if average == 'weighted':
				weights = [supports[c] for c in classes]
				if np.sum(weights) == 0:
					return 0
			return np.average(accuracies, weights=weights)

	def score(self, Y_hat, Y):
		return self.calculate(Y_hat, Y, average='micro')

class Precision(Metric):
	def __init__(self):
		self.name = "precision"

	def calculate(self, Y_hat_, Y, average=None):
		pass


class Recall(Metric):
	def __init__(self):
		self.name = "recall"

	def calculate(self, Y_hat_, Y, average=None):
		pass


class FScore(Metric):
	def __init__(self, beta=1):
		self.name = "f-score"
		self.beta = beta

	def calculate(self, Y_hat_, Y, average=None):
		pass


class ROCAUC(Metric):
	def __init__(self):
		self.name = 'roc-auc'

	def calculate(self, Y_hat_, Y, average=None):
		pass
