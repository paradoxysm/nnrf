import numpy as np
from sklearn.neighbors import NearestNeighbors

from nnrf.utils import check_XY, one_hot, decode
from nnrf.analysis import get_metrics

from nnrf.utils._estimator import BaseEstimator


class DESKNN(BaseEstimator):
	"""
	Neural Network structured as a Decision Tree.

	Parameters
	----------
	ensemble : BaseEstimator
		Ensemble to use for dynamic selection.
		Needs to be trained.

	k : int, 100
		Parameter for k-nearest neighbors. Query
		regions gather `k` nearest neighbors to
		rank ensemble estimators.

	leaf_size : int, default=40
		Number of points at which to switch to brute-force.
		Changing leaf_size will not affect the results of a query,
		but can significantly impact the speed of a query and
		the memory required to store the constructed tree.

	selection : None, int, or float, default=None
		Determines how selection is operated.
		Must be one of:
		 - None : Use all estimators.
		 - int : Use the top `selection` estimators.
		 - float : Use the top `selection * n_estimators` estimators.

	rank : bool, default=True
		Determines if estimators are weighted
		by the rank, calculated as their score
		in the query region.

	verbose : int, default=0
		Verbosity of estimator; higher values result in
		more verbose output.

	warm_start : bool, default=False
		Determines warm starting to allow training to pick
		up from previous training sessions.

	metric : str, Metric, or None, default='accuracy'
		Metric for estimator ranking. If None, uses
		accuracy.

	Attributes
	----------
	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.

	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.

	data_ : ndarray, shape=(n_samples, n_features)
		Training data.

	targets_ : ndarray, shape(n_samples,)
		Labels for training data.

	scores_ : ndarray, shape=(n_samples, n_estimators)
		Accuracy scores for each estimator in the ensemble.
		1 if the estimator was correct, 0 otherwise.
	"""
	def __init__(self, ensemble, k=100, leaf_size=40,
					selection=None, rank=True, verbose=0, warm_start=False,
					metric='accuracy'):
		super().__init__(verbose=verbose, warm_start=warm_start,
							metric=metric)
		self.ensemble = ensemble
		self.k = k
		self.selection = selection
		self.rank = rank
		self.knn = NearestNeighbors(n_neighbors=k, leaf_size=leaf_size)
		self.data_ = np.array([])
		self.targets_ = np.array([])
		self.scores_ = np.array([])

	def fit(self, X, Y, weights=None):
		"""
		Train the model on the given data and labels.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Training data.

		Y : array-like, shape=(n_samples,)
			Target labels as integers.

		weights : array-like, shape=(n_samples,), default=None
			Sample weights for ensemble.
			If None, then samples are equally weighted.

		Returns
		-------
		self : Base
			Fitted estimator.
		"""
		X, Y = check_XY(X=X, Y=Y)
		if not self.ensemble._is_fitted():
			raise ValueError("Ensemble must already be trained")
		if X.shape[1] != self.ensemble.n_features_:
			raise ValueError("Ensemble accepts data with %d features" % \
								self.ensemble.n_features_,
								"but encountered data with %d features." % \
								X.shape[1])
		if self.verbose > 0 : print("Initializing and training model")
		if self.n_classes_ is None : self.n_classes_ = self.ensemble.n_classes_
		if self.n_features_ is None : self.n_features_ = self.ensemble.n_features_
		self.data_, self.targets_ = X, Y.reshape(-1)
		if self.verbose > 2 : print("Fitting Nearest Neighbors")
		self.knn.fit(X)
		if self.verbose > 2 : print("Scoring ensemble")
		for e in self.ensemble.estimators_:
			p = np.where(e.predict(X) == Y.reshape(-1), 1, 0)
			self.scores_ = np.concatenate((self.scores_, p))
		self.scores_ = self.scores_.reshape(-1, len(X)).T
		self.fitted_ = True
		if self.verbose > 0 : print("Training complete.")
		return self

	def predict_proba(self, X):
		"""
		Predict class probabilities for each sample in `X`.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		proba : array-like, shape=(n_samples, n_classes)
			Class probabilities of input data.
			The order of classes is in sorted ascending order.
		"""
		if not self._is_fitted():
			raise RunTimeError("Model is not fitted")
		X = check_XY(X=X)
		if self.verbose > 0 : print("Predicting %d samples." % \
								X.shape[0])
		d, i = self.knn.kneighbors(X)
		competence = self._calculate_competence(i) #NE
		i_estimators = self._select(competence) # Ne
		estimators = self.ensemble.estimators_
		n_estimators = len(estimators)
		pred = np.zeros((len(X), self.n_classes_))
		for n in range(len(X)):
			for e in range(n_estimators):
				estimator = estimators[i_estimators[n, e]]
				p = estimator.predict_proba(X[n].reshape(1,-1)) # NC
				if self.rank : p *= competence[n, i_estimators[n, e]]
				pred[n] += p.reshape(-1)
		return pred / n_estimators

	def _is_fitted(self):
		"""
		Return True if the model is properly ready
		for prediction.

		Returns
		-------
		fitted : bool
			True if the model can be used to predict data.
		"""
		data = len(self.data_) > 0
		targets = len(self.targets_) > 0
		score = len(self.scores_) > 0
		ensemble = self.ensemble._is_fitted()
		return data and targets and score and ensemble and self.fitted_

	def _calculate_competence(self, i):
		"""
		Calculate competence scores of estimators
		in the ensemble based on the metric.

		Parameters
		----------
		i : array-like, shape=(n_samples, n_neighbors)
			Indices of data to calculate competence
			for each data sample.

		Returns
		-------
		competence : ndarray, shape=(n_sample, n_estimators)
			Scores of each estimator for the given data.
		"""
		if self.metric.name == 'accuracy':
			s = self.scores_[i] #NIE
			return np.mean(s, axis=1) # NE
		competence = []
		X, Y = self.data_[i], self.targets_[i] # NIF, NI
		for e in self.ensemble.estimators_:
			Y_hat = e.predict(X.reshape(-1, X.shape[-1]))
			Y_hat = Y_hat.reshape(X.shape[0],-1) # NI
			c = [self.metric.score(Y_hat[k], Y[k]) for k in len(Y)] # N
			competence.append(c)
		return np.array(competence).T

	def _select(self, competence):
		"""
		Select estimators to use for classification.

		Parameters
		----------
		competence : ndarray, shape=(n_samples, n_estimators)
			Scores of each estimator for the given data.

		Returns
		-------
		estimators : ndarray, shape=(n_samples, n_selected)
			Indices of selected estimators.
		"""
		n = len(self.ensemble.estimators_)
		if self.selection is None:
			return np.array([np.arange(n) for i in range(len(competence))])
		elif isinstance(self.selection, int) and 0 < self.selection <= n:
			return np.argpartition(competence, - self.selection)[:, - self.selection:]
		elif isinstance(self.selection, float) and 0 < self.selection <= 1:
			selection = int(self.selection * n)
			return np.argpartition(competence, - selection)[:, - selection:]
