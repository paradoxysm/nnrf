import numpy as np

def is_float(string):
	try:
		float(string)
		return True
	except ValueError:
		return False

def is_int(string):
	try:
		int(string)
		return True
	except ValueError:
		return False

def check_XY(X=None, Y=None):
	if X is not None and Y is not None:
		if X.shape[0] != Y.shape[0]:
			raise ValueError("X and Y do not have the same length")
		return X, Y
	elif X is not None:
		return check_X(X)
	elif Y is not None:
		return check_X(Y)

def check_X(X):
	X = np.array(X)
	if X.shape[0] == X.size:
		X = X.reshape(-1,1)
	return X
