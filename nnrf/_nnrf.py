import numpy as np
from tqdm import trange

from nnrf import NNDT, NeuralNetwork
from nnrf.ml import get_loss, get_activation, get_regularizer
from nnrf.ml.activation import PReLU
from nnrf.utils import check_XY, one_hot, calculate_batch, \
						calculate_bootstrap, \
						create_random_state, BatchDataset
from nnrf.analysis import get_metrics

from nnrf._estimator import BaseClassifier


class NNRF(BaseClassifier):
	"""
	Neural Network structured as a Decision Tree.

	Parameters
	----------
	n : int, default=50
		Number of base estimators (NNDTs) in the random forest.

	d : int, default=5
		Depth, or number of layers, of the NNDT.

	r : int, float, None, or {'sqrt', 'log2'}, default='sqrt'
		The number of features provided as input to each
		node. Must be one of:
		 - int : Use `r` features.
		 - float : Use `r * n_features` features.
		 - 'sqrt' : Use `sqrt(n_features)` features.
		 - 'log2' : Use `log2(n_features)` features.
		 - None : Use all features.

	loss : str, LossFunction, default='cross-entropy'
		Loss function to use for training. Must be
		one of the default loss functions or an object
		that extends LossFunction.

	activation : str, Activation, default=PReLU(0.2)
		Activation function to use at each node in the NNDT.
		Must be one of the default loss functions or an
		object that extends Activation.

	regularize : str, Regularizer, None, default=None
		Regularization function to use at each node in the NNDT.
		Must be one of the default regularizers or an object that
		extends Regularizer. If None, no regularization is done.

	alpha : float, default=0.001
		Learning rate. NNDT uses gradient descent.

	max_iter : int, default=10
		Maximum number of epochs to conduct during training.

	tol : float, default=1e-4
		Convergence criteria for early stopping.

	bootstrap_size : int, float, default=None
		Bootstrap size for training. Must be one of:
		 - int : Use `bootstrap_size`.
		 - float : Use `bootstrap_size * n_samples`.
		 - None : Use `n_samples`.

	batch_size : int, float, default=None
		Batch size for training. Must be one of:
		 - int : Use `batch_size`.
		 - float : Use `batch_size * n_samples`.
		 - None : Use `n_samples`.

	class_weight : dict, 'balanced', or None, default=None
		Weights associated with classes in the form
		`{class_label: weight}`. Must be one of:
		 - None : All classes have a weight of one.
		 - 'balanced': Class weights are automatically calculated as
						`n_samples / (n_samples * np.bincount(Y))`.

	verbose : int, default=0
		Verbosity of estimator; higher values result in
		more verbose output.

	warm_start : bool, default=False
		Determines warm starting to allow training to pick
		up from previous training sessions.

	metric : str, Metric, or None, default='accuracy'
		Metric for estimator score.

	random_state : None or int or RandomState, default=None
		Initial seed for the RandomState. If seed is None,
		return the RandomState singleton. If seed is an int,
		return a RandomState with the seed set to the int.
		If seed is a RandomState, return that RandomState.

	Attributes
	----------
	estimators_ : list of NNDT, shape=(n_estimators_)
		List of all estimators in the NNRF.

	n_classes_ : int
		Number of classes.

	n_features_ : int
		Number of features.

	fitted_ : bool
		True if the model has been deemed trained and
		ready to predict new data.
	"""
	def __init__(self, n=50, d=5, r='sqrt', alpha=0.001, loss='cross-entropy',
					activation=PReLU(0.2), regularize=None, max_iter=10, tol=1e-4,
					bootstrap_size=None, batch_size=None, class_weight=None,
					verbose=0, warm_start=False, metric='accuracy',
					random_state=None):
		super().__init__(loss=loss, max_iter=max_iter, tol=tol,
						batch_size=batch_size, verbose=verbose,
						warm_start=warm_start, class_weight=class_weight,
						metric=metric, random_state=random_state)
		self.n = n
		self.d = d
		self.r = r
		self.alpha = alpha
		self.activation = get_activation(activation)
		self.regularizer = get_regularizer(regularize)
		self.max_iter = max_iter
		self.tol = tol
		self.bootstrap_size = bootstrap_size
		self.estimators_ = []

	def _initialize(self):
		"""
		Initialize the parameters of the NNRF.
		"""
		random_state = self.random_state.choice(self.n*1000, size=self.n, replace=False)
		for n in range(self.n):
			if self.verbose == 0 : verbose = 0
			else : verbose = self.verbose - 1
			nndt = NNDT(d=self.d, r=self.r, alpha=self.alpha, loss=self.loss,
							activation=self.activation,regularize=self.regularizer,
							max_iter=self.max_iter, tol=self.tol,
							batch_size=self.batch_size,
							class_weight=self.class_weight,
							verbose=verbose, warm_start=self.warm_start,
							metric=self.metric,
							random_state=random_state[n])
			nndt.n_classes_ = self.n_classes_
			nndt.n_features_ = self.n_features_
			self.estimators.append(nndt)

	def decision_path(self, X, full=False):
		"""
		Returns the nodes in each layer that comprises
		the decision path of the NNDT.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		full : bool, default=False
			Determines if full paths of all estimators
			are given. If False, return the mean
			path activations.

		Returns
		-------
		paths : ndarray, shape=(n_samples, n_layers) or
					(n_estimators, n_samples, n_layers)
			List of paths for each data sample.
			Each path consists of a list of nodes active
			in each layer (e.g. if the third element is
			2, the third node in the third layer was active).
			If `full` is True, return paths for each estimator.
			If `full` is False, return the mean path activations.
		"""
		paths = []
		for e in self.estimators_:
			paths.append(e.decision_path(X))
		paths = np.array(paths)
		if not full:
			paths = np.mean(paths, axis=0)
		return paths

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
			Sample weights. If None, then samples are equally weighted.

		Returns
		-------
		self : Base
			Fitted estimator.
		"""
		X, Y = check_XY(X=X, Y=Y)
		self.n_classes_ = len(set(Y))
		self.n_features_ = X.shape[1]
		try : Y = one_hot(Y, cols=self.n_classes_)
		except : raise
		bootstrap = calculate_bootstrap(self.bootstrap_size, len(X))
		batch_size = calculate_batch(self.batch_size, len(Y))
		if not self.warm_start or not self._is_fitted():
			if verbose > 0 : print("Initializing model")
			self._initialize()
		if verbose > 0 : print("Training model for %d epochs" % self.max_iter,
								"on %d samples in batches of %d." % \
								(X.shape[0], self.batch_size),
								"Convergence tolerance set to %.4f." % self.tol)
		if verbose > 0 : print("Training model with %d estimators." % self.n)
		estimators = range(len(self.estimators_))
		if verbose == 1 : estimators = trange(len(self.estimators_))
		for e in estimators:
			if verbose == 1 : estimators.set_description("Estimator %d" % e+1)
			elif verbose > 1 : print("Fitting estimator %d" % e+1)
			data = np.concatenate((X, Y, weights.reshape(-1,1)), axis=1)
			data = self.random_state.permutation(data)
			X_ = data[:bootstrap, :X.shape[1]]
			Y_ = data[:bootstrap, X.shape[1]:-1]
			weights = data[:bootstrap, -1]
			e.fit(X_, Y_, weights=weights)
		self.fitted_ = True
		if verbose > 0 : print("Training complete.")
		return self

	def set_warm_start(self, warm):
		"""
		Set the `warm_start` attribute of
		all estimators to `warm`.

		Parameters
		----------
		warm : bool
			Status to set all estimators.
		"""
		for e in self.estimators_:
			e.warm_start = warm

	def _is_fitted(self):
		"""
		Return True if the model is properly ready
		for prediction.

		Returns
		-------
		fitted : bool
			True if the model can be used to predict data.
		"""
		estimators = len(self.estimators_) > 0
		return estimators and self.fitted_

	def _forward(self, X):
		"""
		Conduct the forward propagation steps through the model.

		Parameters
		----------
		X : array-like, shape=(n_samples, n_features)
			Data to predict.

		Returns
		-------
		Y_hat : array-like, shape=(n_samples, n_classes)
			Output.
		"""
		pred = []
		for e in self.estimators_:
			pred.append(e.predict_proba(X))
		pred = np.array(pred)
		return np.mean(pred, axis=0)
