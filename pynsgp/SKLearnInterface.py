from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import inspect

from genepro.node_impl import *

from pynsgp.Fitness.FitnessFunction import SurvivalRegressionFitness
from pynsgp.Evolution.Evolution import pyNSGP


class pyNSGPEstimator(BaseEstimator, RegressorMixin):

	def __init__(self,
		path,
		X_train,
		y_train,
		X_test,
		y_test,
		crossovers,
		mutations,
		coeff_opts,
		pop_size=100,
		max_generations=100, 
		max_evaluations=-1,
		max_time=-1,
		functions=[Plus(), Minus(), Times(), Div()],
		use_erc=True,
		error_metric='cindex_ipcw',
		size_metric='distinct_raw_features',
		prob_delete_tree=0.05,
		prob_init_tree=0.1,
		prob_mt_crossover=0.8,
		initialization_max_tree_height=4,
		min_depth=2,
		tournament_size=4,
		max_tree_size=100,
		partition_features=False,
		min_trees_init=1,
		max_trees_init=5,
		penalize_duplicates=True,
		verbose=False,
		alpha=0.01,
		n_iter=100,
		l1_ratio=0.9
		):

		self.largest_value = 1e+8

		args, _, _, values = inspect.getargvalues(inspect.currentframe())
		values.pop('self')
		for arg, val in values.items():
			setattr(self, arg, val)


	def fit(self, X, y):

		# Check that X and y have correct shape
		X_train, y_train = check_X_y(self.X_train, self.y_train)
		self.X_ = X_train
		self.y_ = y_train

		X_test, y_test = check_X_y(self.X_test, self.y_test)

		fitness_function = SurvivalRegressionFitness(
			X_train,
			y_train,
			metric=self.error_metric,
			size_proxy=self.size_metric,
			alpha=self.alpha,
			n_iter=self.n_iter,
			l1_ratio=self.l1_ratio
		)

		test_fitness_function = SurvivalRegressionFitness(
			X_train,
			y_train,
			metric=self.error_metric,
			size_proxy=self.size_metric,
			alpha=self.alpha,
			n_iter=self.n_iter,
			l1_ratio=self.l1_ratio,
			X_test=X_test,
			y_test=y_test
		)
		
		terminals = []
		if self.use_erc:
			terminals.append(InstantiableConstant())
		n_features = X_train.shape[1]
		self.n_features_in_ = n_features
		for i in range(n_features):
			terminals.append(OOHRdyFeature(i))

		nsgp = pyNSGP(
			path=self.path,
			fitness_function=fitness_function,
			test_fitness_function=test_fitness_function,
			functions=self.functions,
			terminals=terminals,
			crossovers=self.crossovers,
			mutations=self.mutations,
			coeff_opts=self.coeff_opts,
			pop_size=self.pop_size,
			prob_delete_tree=self.prob_delete_tree,
			prob_init_tree=self.prob_init_tree,
			prob_mt_crossover=self.prob_mt_crossover,
			max_generations=self.max_generations,
			max_time=self.max_time,
			max_evaluations=self.max_evaluations,
			initialization_max_tree_height=self.initialization_max_tree_height,
			min_depth=self.min_depth,
			max_tree_size=self.max_tree_size,
			tournament_size=self.tournament_size,
			penalize_duplicates=self.penalize_duplicates,
			verbose=self.verbose,
			partition_features=self.partition_features,
			min_trees_init=self.min_trees_init,
			max_trees_init=self.max_trees_init
		)

		nsgp.Run()
		self.nsgp_ = nsgp

		return self

	def predict(self, X):
		# Check fit has been called
		check_is_fitted(self, ['nsgp_'])

		# Input validation
		X = check_array(X)
		fifu = self.nsgp_.fitness_function
		prediction = fifu.elite(X)
		prediction.clip(-self.largest_value, self.largest_value, out=prediction)

		prediction = fifu.elite.scaler.transform(prediction)

		return prediction

	def score(self, X, y=None):
		if y is None:
			raise ValueError('The ground truth y was not set')
		
		# Check fit has been called
		prediction = self.predict(X)

		fifu = self.nsgp_.fitness_function

		return fifu.elite.cox.score(X=prediction, y=y)

	def get_params(self, deep=True):
		attributes = inspect.getmembers(self, lambda a:not(inspect.isroutine(a)))
		attributes = [a for a in attributes if not (a[0].endswith('_') or a[0].startswith('_'))]

		dic = {}
		for a in attributes:
			dic[a[0]] = a[1]

		return dic

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def get_elitist_obj1(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.fitness_function.elite

	def get_front(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.latest_front

	def get_population(self):
		check_is_fitted(self, ['nsgp_'])
		return self.nsgp_.population

	# string representation: front + objectives
	def __str__(self):
		front = sorted(self.get_front(), key=lambda x: -x.objectives[0])
		already_seen = set()
		result = "ERROR\tCOMPLEXITY\tMODEL\n"
		result += '========================================\n'
		for solution in front:
			string_repr = solution.get_readable_repr()
			if string_repr in already_seen:
				continue
			already_seen.add(string_repr)
			result += "{:.3f}\t{:.3f}\t".format(solution.objectives[0], solution.objectives[1]) 
			result += solution.get_readable_repr()
			result += "\n"
		return result
