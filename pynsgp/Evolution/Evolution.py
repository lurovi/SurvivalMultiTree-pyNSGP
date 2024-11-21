import numpy as np
from numpy.random import random, randint
import time
from copy import deepcopy

from pynsgp.Variation import Variation
from pynsgp.Selection import Selection


class pyNSGP:

	def __init__(
		self,
		fitness_function,
		functions,
		terminals,
		crossovers,
		mutations,
		coeff_opts,
		pop_size=500,
		prob_delete_tree=0.05,
		prob_init_tree=0.1,
		prob_mt_crossover=0.8,
		max_evaluations=-1,
		max_generations=-1,
		max_time=-1,
		initialization_max_tree_height=4,
		min_depth=2,
		max_tree_size=100,
		tournament_size=4,
		penalize_duplicates=True,
		verbose=False,
		partition_features=False,
		min_trees_init=1,
		max_trees_init=5
		):

		self.pop_size = pop_size
		self.fitness_function = fitness_function
		self.functions = functions
		self.terminals = terminals
		self.crossovers = crossovers
		self.mutations = mutations
		self.coeff_opts = coeff_opts
		self.prob_delete_tree = prob_delete_tree
		self.prob_init_tree = prob_init_tree
		self.prob_mt_crossover = prob_mt_crossover

		self.max_evaluations = max_evaluations
		self.max_generations = max_generations
		self.max_time = max_time

		self.initialization_max_tree_height = initialization_max_tree_height
		self.min_depth = min_depth
		self.max_tree_size = max_tree_size
		self.tournament_size = tournament_size
		self.penalize_duplicates = penalize_duplicates

		self.generations = 0

		self.verbose = verbose
		
		self.partition_features = partition_features
		self.min_trees_init = min_trees_init
		self.max_trees_init = max_trees_init
	

	def __ShouldTerminate(self):
		must_terminate = False
		elapsed_time = time.time() - self.start_time
		if self.max_evaluations > 0 and self.fitness_function.evaluations >= self.max_evaluations:
			must_terminate = True
		elif self.max_generations > 0 and self.generations >= self.max_generations:
			must_terminate = True
		elif self.max_time > 0 and elapsed_time >= self.max_time:
			must_terminate = True

		if must_terminate and self.verbose:
			print('Terminating at\n\t', 
				self.generations, 'generations\n\t', self.fitness_function.evaluations, 'evaluations\n\t', np.round(elapsed_time,4), 'seconds')

		return must_terminate


	def Run(self):

		self.start_time = time.time()

		self.population = []

		# ramped half-n-half
		curr_max_depth = self.min_depth
		init_depth_interval = self.pop_size / (self.initialization_max_tree_height - self.min_depth + 1)
		next_depth_interval = init_depth_interval

		for i in range( self.pop_size ):
			if i >= next_depth_interval:
				next_depth_interval += init_depth_interval
				curr_max_depth += 1

			t = Variation.GenerateRandomMultitree(self.functions, self.terminals, max_depth=curr_max_depth,
													partition_features=self.partition_features,
													min_trees_init=self.min_trees_init,
													max_trees_init=self.max_trees_init
												)
			self.fitness_function.Evaluate( t )
			self.population.append( t )



		while not self.__ShouldTerminate():

			selected = Selection.TournamentSelect( self.population, self.pop_size, tournament_size=self.tournament_size )

			O = []
			for i in range( self.pop_size ):
				o = deepcopy(selected[i])
				o = Variation.GenerateOffspringMultitree(
					parent_mt=o,
					crossovers=self.crossovers,
					mutations=self.mutations,
					coeff_opts=self.coeff_opts,
					donors=selected,
					internal_nodes=self.functions,
					leaf_nodes=self.terminals,
					max_depth=self.initialization_max_tree_height,
					constraints= {"max_tree_size": self.max_tree_size},
					partition_features=self.partition_features,
					prob_delete_tree=self.prob_delete_tree,
					prob_init_tree=self.prob_init_tree,
					prob_mt_crossover=self.prob_mt_crossover,
					perform_only_one_op=True
				)

				for single_int_tree_index in range(o.number_of_trees() - 1, -1, -1):
					single_internal_tree = o.trees[single_int_tree_index]
					if (len(single_internal_tree.get_subtree()) > self.max_tree_size) or (single_internal_tree.get_height() < self.min_depth):
						del o.trees[single_int_tree_index]

				if o.number_of_trees() == 0:
					o = deepcopy( selected[i] )
				else:
					self.fitness_function.Evaluate(o)

				O.append(o)


			PO = self.population+O
			
			new_population = []
			fronts = self.FastNonDominatedSorting(PO)
			self.latest_front = deepcopy(fronts[0])

			curr_front_idx = 0
			while curr_front_idx < len(fronts) and len(fronts[curr_front_idx]) + len(new_population) <= self.pop_size:
				self.ComputeCrowdingDistances( fronts[curr_front_idx] )
				for p in fronts[curr_front_idx]:
					new_population.append(p)
				curr_front_idx += 1

			if len(new_population) < self.pop_size:
				# fill in remaining
				self.ComputeCrowdingDistances( fronts[curr_front_idx] )
				fronts[curr_front_idx].sort(key=lambda x: x.crowding_distance, reverse=True) 

				while len(fronts[curr_front_idx]) > 0 and len(new_population) < self.pop_size:
					new_population.append( fronts[curr_front_idx][0] )	# pop first because they were sorted in desc order
					fronts[curr_front_idx].pop(0)

				# clean up leftovers
				while len(fronts[curr_front_idx]) > 0:
					del fronts[curr_front_idx][0]

			self.population = new_population

			self.generations = self.generations + 1

			if self.verbose:
				print ('g:',self.generations,'elite obj1:', np.round(self.fitness_function.elite.objectives[0],4), ', obj2:', np.round(self.fitness_function.elite.objectives[1],4), ', size:', len(self.fitness_function.elite), ', n_trees:', self.fitness_function.elite.number_of_trees())


	def FastNonDominatedSorting(self, population):
		rank_counter = 0
		nondominated_fronts = []
		dominated_individuals = {}
		domination_counts = {}
		current_front = []

		for i in range( len(population) ):
			p = population[i]

			dominated_individuals[p] = []
			domination_counts[p] = 0

			for j in range( len(population) ):
				if i == j:
					continue
				q = population[j]

				if p.dominates(q):
					dominated_individuals[p].append(q)
				elif q.dominates(p):
					domination_counts[p] += 1

			if domination_counts[p] == 0:
				p.rank = rank_counter
				current_front.append(p)

		while len(current_front) > 0:
			next_front = []
			for p in current_front:
				for q in dominated_individuals[p]:
					domination_counts[q] -= 1
					if domination_counts[q] == 0:
						q.rank = rank_counter + 1
						next_front.append(q)
			nondominated_fronts.append(current_front)
			rank_counter += 1
			current_front = next_front

		if self.penalize_duplicates:
			already_seen = set()
			discard_front = []
			sorted_pop = sorted(population, key=lambda x: x.rank)
			for p in sorted_pop: 
				summarized_representation = p.cached_output
				if summarized_representation not in already_seen:
					already_seen.add(summarized_representation)
				else:
					# find p and remove it from its front
					for i, q in enumerate(nondominated_fronts[p.rank]):
						if nondominated_fronts[p.rank][i] == p:
							nondominated_fronts[p.rank].pop(i)
							break
					p.rank = np.inf
					discard_front.append(p)
			if len(discard_front) > 0:
				nondominated_fronts.append(discard_front)
				# fix potentially-now-empty fronts
				nondominated_fronts = [front for front in nondominated_fronts if len(front) > 0]


		return nondominated_fronts


	def ComputeCrowdingDistances(self, front):
		number_of_objs = len(front[0].objectives)
		front_size = len(front)

		for p in front:
			p.crowding_distance = 0

		for i in range(number_of_objs):
			front.sort(key=lambda x: x.objectives[i], reverse=False)

			front[0].crowding_distance = front[-1].crowding_distance = np.inf

			min_obj = front[0].objectives[i]
			max_obj = front[-1].objectives[i]

			if min_obj == max_obj:
				continue

			for j in range(1, front_size - 1):

				if np.isinf(front[j].crowding_distance):
					# if extrema from previous sorting
					continue

				prev_obj = front[j-1].objectives[i]
				next_obj = front[j+1].objectives[i]

				front[j].crowding_distance += (next_obj - prev_obj)/(max_obj - min_obj)
