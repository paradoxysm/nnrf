import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris

from nnrf import NNRF

dataset = load_breast_cancer()
data = dataset['data']
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
min, max = data.min(axis=0), data.max(axis=0)
data = (data - min) / (max - min)
target = dataset['target']
partition = int(0.8 * len(data))
train_X_bc = data[:partition]
train_Y_bc = target[:partition]
test_X_bc = data[partition:]
test_Y_bc = target[partition:]

dataset = load_iris()
data = dataset['data']
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
min, max = data.min(axis=0), data.max(axis=0)
data = (data - min) / (max - min)
target = dataset['target']
partition = int(0.8 * len(data))
train_X_iris = data[:partition]
train_Y_iris = target[:partition]
test_X_iris = data[partition:]
test_Y_iris = target[partition:]

@pytest.mark.parametrize("params", [
	({'n':3}),
	({'n':3, 'd':3, 'r':'log2'}),
	({'n':3, 'loss':'mse'}),
	({'n':3, 'optimizer':'sgd'}),
	({'n':3, 'regularize':'l2'})
])

class TestNNRF:
	def test_nnrf_binary(self, params):
		nnrf = NNRF(**params)
		nnrf.fit(train_X_bc, train_Y_bc)

	def test_nnrf_multi(self, params):
		nnrf = NNRF(**params)
		nnrf.fit(train_X_iris, train_Y_iris)

	def test_nnrf_predict_binary(self, params):
		nnrf = NNRF(**params)
		nnrf.fit(train_X_bc, train_Y_bc)
		nnrf.predict(test_X_bc)

	def test_nnrf_predict_multi(self, params):
		nnrf = NNRF(**params)
		nnrf.fit(train_X_iris, train_Y_iris)
		nnrf.predict(test_X_iris)

def test_nnrf_unfit():
	nnrf = NNRF()
	with pytest.raises(RuntimeError):
		nnrf.predict(test_X_bc)
