# TSP
from random import sample
from numpy import median
from numpy import std

from matplotlib import pyplot as plt

class TSP():
	def __init__(self, cities):
		# Convention: p_variable is a parameter
		self.cities = cities
		self.n = len(cities)
		self.p_children = 2
		self.p_maxIters = 200
		self.numIters = 0
		# The top 1/p_children of a population survive
		# The remaining S survivors produce p_children * S children

		self.p_population = self.p_children * 50
		self.p_mutationRate = 1

		self.tours = self.initialize_population()
		self.init_mean, self.init_stdev = self.average_fitness()
		self.best = None

	def plot_best(self, saveFile=None):
		if self.best is None:
			return
		xs = [x for (x,y) in self.best]
		ys = [y for (x,y) in self.best]
		xs.append(xs[0])
		ys.append(ys[0])

		plt.scatter(xs, ys)
		plt.plot(xs, ys, '-o')
		plt.xticks(range(0,22,2))
		plt.yticks(range(0,22,2))
		plt.title("Best:"+str(self.eval_cities(self.best)))
		if saveFile is not None:
			plt.savefig(saveFile)
		else:
			plt.show()

	def initialize_population(self):
		return [sample(self.cities, self.n) for i in range(self.p_population)]

	def _dist(self, c1, c2):
		(x1, y1) = c1
		(x2, y2) = c2
		return ((x1-x2)**2 + (y1-y2)**2)**.5

	def eval_cities(self, tour):
		distances = [self._dist(tour[i], tour[i-1]) for i,_ in enumerate(tour)]
		return sum(distances)

	def average_fitness(self):
		distances = [self.eval_cities(p) for p in self.tours]
		average = sum(distances)/self.p_population
		stdev = std(distances)
		return average, stdev

	def selection_median(self):
		distances = [self.eval_cities(p) for p in self.tours]
		# This works only if p_children is 2!!! JK
		med_dist = median(distances)
		return [p for (p,d) in zip(self.tours, distances) if d<=med_dist]

	def _swap_mutation(self, tour):
		for iter in range(self.p_mutationRate):
			i,j = sample(xrange(self.n), 2)
			tour[i], tour[j] = tour[j], tour[i]
		return tour

	def mutate_all_children(self, children):
		return [self._swap_mutation(c) for c in children]

	def mutate_trivial(self, children):
		return children

	def alleles_contiguous_rand(self, dad, mom):
		# Also worth writing crossover_contiguous_half
		# For enforcing that exactly half of the chromosomes pass on
		# Also worth trying a markov chain. 80% keep next allele
		y1, y2 = sample(xrange(self.n), 2)
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

	def mating_random(self, parents):
		random_indices = [sample(xrange(len(parents)), 2) \
						for i in range(self.p_population)]
		children = [self.alleles_contiguous_rand(parents[x], parents[y]) \
						for (x,y) in random_indices]
		return children

	def termination_fixed_iters(self):
		if self.numIters >= self.p_maxIters:
			return False
		self.numIters += 1
		return True

	def termination_stdev(self):
		multiplier = 0.001
		avg, stdev = self.average_fitness()
		if self.init_stdev * multiplier > stdev or self.numIters >= self.p_maxIters:
			return False
		self.numIters += 1
		return True

	def evolve_naive(self):
		while self.termination_stdev():
			survivors = self.selection_median()
			children = self.mating_random(survivors)
			#mutated = self.mutate_all_children(children)
			mutated = self.mutate_trivial(children)
			self.tours = mutated
			print "Iteration", self.numIters, ":", self.average_fitness()
		self.best = min(self.tours, key=lambda x: self.eval_cities(x))

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


# evolve_naive Test case

cities = [(2,2), (2,4), (6,8), (4,12), (2,16), (6,20), (8, 18), (10,16), (10,12), (14,18), (18,20), (20,16), (18,10), (14,14), (12,8), (18,6), (20,4), (16,2), (10,4), (6,2)]
numIters = 20

for i in range(numIters):
	tsp = TSP(cities)
	tsp.evolve_naive()
	tsp.plot_best(saveFile="./path"+str(i))
	plt.cla()
	print "-------------------------------------"

# _crossover Test case

dad = range(1,10,1)
mom = range(9,0,-1)
dad_indices = [6,7,8]


"""

What are other crossover techniques for TSP?

"""


