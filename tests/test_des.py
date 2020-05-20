import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris

from nnrf import DESKNN, NNRF

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
partition = int(0.7 * len(train_X_bc))
train_X_bc, stack_X_bc = train_X_bc[:partition], train_X_bc[partition:]
train_Y_bc, stack_Y_bc = train_Y_bc[:partition], train_Y_bc[partition:]

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
partition = int(0.7 * len(train_X_iris))
train_X_iris, stack_X_iris = train_X_iris[:partition], train_X_iris[partition:]
train_Y_iris, stack_Y_iris = train_Y_iris[:partition], train_Y_iris[partition:]

@pytest.fixture
def nnrf_binary():
	return NNRF(n=3).fit(train_X_bc, train_Y_bc)

@pytest.fixture
def nnrf_multi():
	return NNRF(n=3).fit(train_X_iris, train_Y_iris)

@pytest.mark.parametrize("params", [
	({}),
	({'selection':1}),
	({'selection':0.5}),
	({'rank':False})
])

class TestDES:
	def test_des_binary(self, nnrf_binary, params):
		des = DESKNN(ensemble=nnrf_binary, k=10, **params)
		des.fit(stack_X_bc, stack_Y_bc)

	def test_nnrf_multi(self, nnrf_multi, params):
		des = DESKNN(ensemble=nnrf_multi, k=10, **params)
		des.fit(stack_X_iris, stack_Y_iris)

	def test_des_predict_binary(self, nnrf_binary, params):
		des = DESKNN(ensemble=nnrf_binary, k=10, **params)
		des.fit(stack_X_bc, stack_Y_bc)
		des.predict(test_X_bc)

	def test_des_predict_multi(self, nnrf_multi, params):
		des = DESKNN(ensemble=nnrf_multi, k=10, **params)
		des.fit(stack_X_iris, stack_Y_iris)
		des.predict(test_X_iris)

def test_des_unfit():
	des = DESKNN(ensemble=NNRF())
	with pytest.raises(RuntimeError):
		des.predict(test_X_bc)
