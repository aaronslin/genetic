# TSP
import random
from numpy import median

class TSP():
	def __init__(self, cities):
		# Convention: p_variable is a parameter
		self.cities = cities
		self.n = len(cities)
		self.p_children = 2
		# The top 1/p_children of a population survive
		# The remaining S survivors produce p_children * S children

		self.p_population = self.p_children * 50
		self.p_mutationRate = 1

	def dist(self, c1, c2):
		(x1, y1) = c1
		(x2, y2) = c2
		return ((x1-x2)**2 + (y1-y2)**2)**.5

	def eval_cities(self, tour):
		distances = [self.dist(tour[i], tour[i-1]) for i,_ in enumerate(tour)]
		return sum(distances)

	def initialize_population(self):
		return [random.shuffle(self.cities) for i in range(self.p_population)]

	def selection_median(self, tours):
		distances = [self.eval_cities(p) for p in tours]
		med_dist = median(distances)
		return [p for (p,d) in zip(tours, distances) if d<med_dist]

	def mutation_swap(self, tour):
		for iter in range(self.p_mutationRate):
			i,j = random.sample(xrange(self.n), 2)
			tour[i], tour[j] = tour[j], tour[i]
		return tour

	def crossover_contiguous_rand(self, dad, mom):
		# Also worth writing crossover_contiguous_half
		# For enforcing that exactly half of the chromosomes pass on
		# Also worth trying a markov chain. 80% keep next allele
		y1, y2 = random.sample(xrange(self.n), 2)
		y1, y2 = min(y1, y2), max(y1, y2)
		return self._crossover(dad, mom, range(y1, y2))

	def _crossover(self, dad, mom, dad_indices):
		dad_cities = {dad[i]: True for i in dad_indices}
		dad_indices = {i: True for i in dad_indices}
		mom_pointer = 0
		child = []

		for i in range(self.n):
			while mom_pointer<self.n and mom[mom_pointer] in dad_cities:
				mom_pointer+=1
			if i in dad_indices:
				child.append(dad[i])
			else:
				child.append(mom[mom_pointer])
				mom_pointer+=1
		return child

	def selection(self, tours):
		# Need some kind of parameter to specify selection criterion
		# Potential for better algorithms than just "top 50% pass"
		pass

	def mutation(self, tour):
		# Need a parameter to specify mutation rate
		# Potential for mutations that aren't just swaps
		pass

	def crossover(self, dad, mom): 
		# How do you choose which segments to crossover?
		# Perhaps there are ways to optimize for segments to crossover
		# Potential ML opportunity to select subsets
		# 	Prediction: will converge to contiguous selections
		pass


	def evolve(self, f_fitness, f_termination):
		# Does this kind of crossover strategy result in a stabel population?
		# 	i.e. in the endgame, are all children "decently good"?
		# Perhaps you mutate one child and don't mutate the other child
		pass


# Test case

dad = range(1,10,1)
mom = range(9,0,-1)
dad_indices = [6,7,8]

tsp = TSP(dad)
print tsp._crossover(dad, mom, dad_indices)



