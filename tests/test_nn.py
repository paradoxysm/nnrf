import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris

from nnrf import NeuralNetwork

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
	({'layers':tuple()}),
	({'layers':(200,100)}),
	({'loss':'mse'}),
	({'optimizer':'sgd'}),
	({'regularize':'l2'})
])

class TestNN:
	def test_nn_binary(self, params):
		nn = NeuralNetwork(**params)
		nn.fit(train_X_bc, train_Y_bc)

	def test_nn_multi(self, params):
		nn = NeuralNetwork(**params)
		nn.fit(train_X_iris, train_Y_iris)

	def test_nn_predict_binary(self, params):
		nn = NeuralNetwork(**params)
		nn.fit(train_X_bc, train_Y_bc)
		nn.predict(test_X_bc)

	def test_nn_predict_multi(self, params):
		nn = NeuralNetwork(**params)
		nn.fit(train_X_iris, train_Y_iris)
		nn.predict(test_X_iris)

def test_nn_unfit():
	nn = NeuralNetwork()
	with pytest.raises(RuntimeError):
		nn.predict(test_X_bc)

def test_nn_importance():
	nn = NeuralNetwork().fit(train_X_iris, train_Y_iris)
	nn.feature_importance(train_X_iris, train_Y_iris)
