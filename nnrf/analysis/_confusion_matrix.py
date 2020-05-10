import numpy as np

def confusion_matrix(Y_hat, Y, norm=None):
		"""
		Calculate confusion matrix.

		Parameters
		----------
		Y_hat : array-like
			List of data labels.

		Y : array-like
			List of target truth labels.

		norm : {'label', 'target', 'all', None}, default=None
			Normalization on resulting matrix. Must be one of:
			 - 'label' : normalize on labels (columns).
			 - 'target' : normalize on targets (rows).
			 - 'all' : normalize on the entire matrix.
			 - None : No normalization.

		Returns
		-------
		matrix : ndarray, shape=(target_classes, label_classes)
			Confusion matrix with target classes as rows and
			label classes as columns. Classes are in sorted order.
		"""
		target_classes = sorted(set(Y))
		label_classes = sorted(set(Y_hat))
		target_dict = {target_classes[k]: k for k in range(len(target_classes))}
		label_dict = {label_classes[k]: k for k in range(len(label_classes))}
		matrix = np.zeros((len(target_classes), len(label_classes)))
		for label, target in zip(Y_hat, Y):
			matrix[target_dict[target],label_dict[label]] += 1
		if norm == 'label':
			matrix /= np.max(matrix, axis=0).reshape((1,matrix.shape[1]))
		elif norm == 'target':
			matrix /= np.max(matrix, axis=1).reshape((matrix.shape[0],1))
		elif norm == 'all':
			matrix /= np.max(matrix)
		elif norm is not None:
			raise ValueError("Norm must be one of {'label', 'target', 'all', None}")
		return matrix.astype(int)

def multiconfusion_matrix(Y_hat, Y):
	"""
	Calculate confusion matrices for each class in `targets`.

	Parameters
	----------
	Y_hat : array-like
		List of data labels.

	Y : array-like
		List of target truth labels.

	Returns
	-------
	matrices : dict of ndarrays, shape=(2,2)
		Dictionary of 2x2 confusion matrices for each class,
		with true values as rows and label values as columns.
		Matrix is as follows:
			0	1
		0	TN	FP
		1	FN	TP
	"""
	target_classes = sorted(set(Y))
	target_dict = {target_classes[k]: k for k in range(len(target_classes))}
	matrices = np.zeros((len(target_classes),2,2))
	for label, target in zip(Y_hat, Y):
		if label == target:
			matrices[target_dict[target],1,1] += 1
			matrices[target_dict[target],0,0] -= 1
			matrices[:,0,0] += 1
		elif label != target and label in target_classes:
			matrices[target_dict[label],0,1] += 1
			matrices[target_dict[target],1,0] += 1
	matrices = {target_classes[k]: matrices[k] for k in range(len(target_classes))}
	return matrices
