import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer

from nnrf import NeuralNetwork

dataset = load_breast_cancer()
data = dataset['data']
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
min, max = data.min(axis=0), data.max(axis=0)
data = (data - min) / (max - min)
target = dataset['target']
partition = int(0.8 * len(data))
train_X = data[:partition]
train_Y = target[:partition]
test_X = data[partition:]
test_Y = target[partition:]

@pytest.mark.parametrize("nn", [
	(NeuralNetwork()),
	(NeuralNetwork(layers=tuple())),
	(NeuralNetwork(layers=(200,100))),
	(NeuralNetwork(regularize='l2'))
])

def test_nn(nn):
	nn.fit(train_X, train_Y)

@pytest.mark.parametrize("nn", [
	(NeuralNetwork()),
	(NeuralNetwork(layers=tuple())),
	(NeuralNetwork(layers=(200,100))),
	(NeuralNetwork(regularize='l2'))
])

def test_nn_predict(nn):
	nn.fit(train_X, train_Y)
	nn.predict(test_X)

def test_nn_unfit():
	nn = NeuralNetwork()
	with pytest.raises(RuntimeError):
		nn.predict(test_X)
