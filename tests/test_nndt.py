import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris

from nnrf import NNDT

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
	({}),
	({'d':3, 'r':'log2'}),
	({'loss':'mse'}),
	({'optimizer':'sgd'}),
	({'regularize':'l2'})
])

class TestNNDT:
	def test_nndt_binary(self, params):
		nndt = NNDT(**params)
		nndt.fit(train_X_bc, train_Y_bc)

	def test_nndt_multi(self, params):
		nndt = NNDT(**params)
		nndt.fit(train_X_iris, train_Y_iris)

	def test_nndt_predict_binary(self, params):
		nndt = NNDT(**params)
		nndt.fit(train_X_bc, train_Y_bc)
		nndt.predict(test_X_bc)

	def test_nndt_predict_multi(self, params):
		nndt = NNDT(**params)
		nndt.fit(train_X_iris, train_Y_iris)
		nndt.predict(test_X_iris)

def test_nndt_unfit():
	nndt = NNDT()
	with pytest.raises(RuntimeError):
		nndt.predict(test_X_bc)
