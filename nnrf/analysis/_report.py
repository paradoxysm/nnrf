import numpy as np

from sleep.utils.analysis import multiconfusion_matrix

def calculate_statistics(labels, targets, beta=1, average=None):
		"""
		Calculate the precisions, recalls, F-beta scores, and
		supports for each class in `targets`.

		Parameters
		----------
		labels : array-like
			List of data labels.

		targets : array-like
			List of target truth labels.

		beta : float, default=1
			Strength of recall relative to precision in the F-score.

		average : {'micro', 'macro', 'weighted', None}, default=None
			The type of averaging to perform on statistics. Must be one of:
			 - None : Do not perform averaging, statistics for each class
			 			are returned.
			 - 'micro' : Calculate globally, counting total true positives,
			 				false negatives, and false positives.
			 - 'macro' : Calculate per class an unweighted mean.
			 - 'weighted' : Calculate per class the mean weighted by support.

		Returns
		-------
		precisions : float or dict
			Dictionary of precisions for each class if `average` is None.
			Averaged precision based on averaging method if provided.

		recalls : float or dict
			Dictionary of recalls for each class if `average` is None.
			Averaged recall based on averaging method if provided.

		fscores : float or dict
			Dictionary of fscores for each class if `average` is None.
			Averaged fscore based on averaging method if provided.

		supports : float or dict
			Dictionary of supports for each class if `average` is None.
			Total support (number of classes) if averaging method is provided.
		"""
		if beta < 0:
			raise ValueError("Beta must be non-negative")
		matrix = multiconfusion_matrix(labels, targets)
		matrix_labels = list(matrix.keys())
		matrix = np.array([matrix[l] for l in matrix_labels])
		tp_sum = matrix[:,1,1]
		label_sum = tp_sum + matrix[:,0,1]
		target_sum = tp_sum + matrix[:,1,0]
		if average == 'micro':
			tp_sum = np.array([tp_sum.sum()])
			label_sum = np.array([label_sum.sum()])
			target_sum = np.array([target_sum.sum()])
		with np.errstate(divide='ignore', invalid='ignore'):
			precisions = np.divide(tp_sum, label_sum,
									out=np.zeros(tp_sum.shape, dtype=float),
									where=label_sum!=0)
			recalls = np.divide(tp_sum, target_sum,
									out=np.zeros(tp_sum.shape, dtype=float),
									where=target_sum!=0)
		if np.isposinf(beta):
			fscores = recalls
		else:
			beta2 = beta ** 2
			denom = beta2 * precisions + recalls
			fscores = (1 + beta2) * precisions * recalls / denom
		if average == 'weighted':
			weights = target_sum
			if target_sum.sum() == 0:
				return 0, 0, 0, target_sum.sum()
		else:
			weights = None
		if average is not None:
			precisions = np.average(precisions, weights=weights)
			recalls = np.average(recalls, weights=weights)
			fscores = np.average(fscores, weights=weights)
			supports = target_sum.sum()
		else:
			precisions = {matrix_labels[k]: precisions[k] for k in range(len(matrix_labels))}
			recalls = {matrix_labels[k]: recalls[k] for k in range(len(matrix_labels))}
			fscores = {matrix_labels[k]: fscores[k] for k in range(len(matrix_labels))}
			supports = {matrix_labels[k]: target_sum[k] for k in range(len(matrix_labels))}
		return precisions, recalls, fscores, supports

def classification_report(labels, targets, beta=1):
	"""
	Create a report on classification statistics.

	Parameters
	----------
	labels : array-like
		List of data labels.

	targets : array-like
		List of target truth labels.

	beta : float, default=1
		Strength of recall relative to precision in the F-score.

	Returns
	-------
	report : dict
		Dictionary containing classification statistics in the following
		structure:
		 - 'label': {
		 				'precision':0.5,
						'recall':1.0,
						'f-score':0.67,
						'support':1
		 			},
		   ...
		 - 'beta': 1,
		 - 'support': 5,
		 - 'accuracy': 0.8,
		 - 'macro avg': {
		 				'precision':0.6,
						'recall':0.9,
						'f-score':0.67,
		 			},
		 - 'weighted avg': {
		 				'precision':0.67,
						'recall':0.9,
						'f-score':0.67,
		 			}
	"""
	stats = calculate_statistics(labels, targets, beta=beta)
	_, _, accuracy, total = calculate_statistics(labels, targets, beta=beta, average='micro')
	macro = calculate_statistics(labels, targets, beta=beta, average='macro')
	weighted = calculate_statistics(labels, targets, beta=beta, average='weighted')
	h = ['precision', 'recall', 'f-score', 'support']
	report = {
				'beta': beta,
				'support': total,
				'accuracy': accuracy,
				'macro avg': {h[i]: macro[i] for i in range(len(h))},
				'weighted avg': {h[i]: weighted[i] for i in range(len(h))}
			}
	classes = set(stats[0].keys())
	for c in classes:
		report[c] = {h[i]: stats[i][c] for i in range(len(h))}
	return report
