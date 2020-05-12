
def vars_recurse(obj):
	"""
	Recursively collect vars() of the object.

	Parameters
	----------
	obj : object
		Object to collect attributes

	Returns
	-------
	params : dict
		Dictionary of object parameters.
	"""
	if hasattr(obj, '__dict__'):
		params = vars(obj)
		for k in params.keys():
			if hasattr(params[k], '__dict__'):
				params[k] = vars_recurse(params[k])
		params = dict(list(params.items()) + [('type', type(obj))])
		return params
	raise ValueError("obj does not have __dict__ attribute")

class Base:
	"""
	Base object class for nnrf.
	"""
	def __init__(self):
		pass

	def get_params(self):
		"""
		Get all parameters of the object, recursively.

		Returns
		-------
		params : dict
			Dictionary of object parameters.
		"""
		params = vars(self)
		for k in params.keys():
			if hasattr(params[k], 'get_params'):
				params[k] = dict(list(k.get_params().items()) + \
								[('type', type(self))])
			elif isinstance(params[k], np.random.RandomState):
				params[k] = {'type': np.random.RandomState,
								'seed': params[k].get_state()}
			elif hasattr(params[k], '__dict__'):
				params[k] = vars_recurse(params[k])
		params = dict(list(params.items()) + [('type', type(self))])
		return params

	def set_params(self, params):
		"""
		Set the attributes of the object with the given
		parameters.

		Parameters
		----------
		params : dict
			Dictionary of object parameters.

		Returns
		-------
		self : Base
			Itself, with parameters set.
		"""
		valid = self.get_params().keys()
		for k, v in params.items():
			if k not in valid:
				raise ValueError("Invalid parameter %s for object %s" % \
									(k, self.__name__))
			param = v
			if isinstance(v, dict) and 'type' in v.keys():
				t = v['type']
				if t == np.random.RandomState:
					state = v['seed']
					param = np.random.RandomState().set_state(state)
				elif 'set_params' in dir(t):
					param = t().set_params(v.pop('type'))
				else:
					param = t()
					for p, p_v in v.pop('type').items():
						setattr(param, p, p_v)
			setattr(self, k, param)
		return self
