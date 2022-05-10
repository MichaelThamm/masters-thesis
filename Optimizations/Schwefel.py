from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from platypus import *
from enum import Enum
import numpy as np
import logging
import copy
import os

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
https://www.indusmic.com/post/schwefel-function
'''

PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Optimizations')


class ExitReason(Enum):
    MAX_EVALS = 1
    OBJECTIVE_TOLERANCE = 2


# Problem is the object that Test will inherit from
class Constraints(Problem):

    def __init__(self, _lower, _upper):
        # The numbers indicate: #inputs, #objectives, #constraints
        super(Constraints, self).__init__(2, 1)
        # Constrain the range and type for each input
        self.types[:] = [Real(_lower, _upper), Real(_lower, _upper)]
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE]

    def evaluate(self, solution):
        x = solution.variables[0]
        y = solution.variables[1]
        magnitude_fitness = schwefel(x, y)
        solution.objectives[:] = [magnitude_fitness]


class WrappedTerminationCondition(TerminationCondition):
    def __init__(self, max_evals, tolerance):
        super(TerminationCondition, self).__init__()
        self.iteration = 0
        self.max_evals = max_evals
        self.solver_tolerance = tolerance
        self.old_fittest = np.inf
        self.reason = None

    def __call__(self, algorithm):
        return self.shouldTerminate(algorithm)

    def initialize(self, algorithm):
        pass

    def shouldTerminate(self, algorithm):
        self.iteration += 1
        if self.iteration > self.max_evals:
            self.reason = ExitReason.MAX_EVALS
            return True
        if hasattr(algorithm, 'fittest'):
            if self.old_fittest - algorithm.fittest.objectives[0] < self.solver_tolerance:
                self.reason = ExitReason.OBJECTIVE_TOLERANCE
                return True
            else:
                self.old_fittest = algorithm.fittest.objectives[0]
                return False
        # The initial check of condition(self) has no algorithm.fittest attribute so we continue until its set
        else:
            return False


def plotResults(str_name, algorithm):
    if not algorithm:
        return
    feasible_solutions = [s for s in unique(nondominated(algorithm.result))]
    plotResults = []
    for results in feasible_solutions:
        x, y = (algorithm.problem.types[0].decode(results.variables[0]), algorithm.problem.types[0].decode(results.variables[1]))
        objective = results.objectives[0]
        # TODO printing algorithm.nfe I dont think is useful unless this .nfe is different than the one in core.py Algorithm
        #  Check if its printing 5000 or something else
        print(f'***{str_name}*** coords: {(x, y)}, objective: {objective}, iterations: {algorithm.nfe}')
        plotResults.append((x, y, objective))

    fig, axs = plt.subplots(3)
    fig.suptitle(str_name)
    axs[0].scatter(range(len(feasible_solutions)), [x[0] for x in plotResults])
    axs[0].set(ylabel='x')
    axs[1].scatter(range(len(feasible_solutions)), [x[1] for x in plotResults])
    axs[1].set(ylabel='y')
    axs[2].scatter(range(len(feasible_solutions)), [x[2] for x in plotResults])
    axs[2].set(ylabel='objective')
    plt.xlabel('Generations')
    plt.show()


def nsgaii(lower, upper, evals, tolerance, population_size, offspring_size, selector, run=False):
    if not run:
        return
    algorithm = NSGAII(Constraints(lower, upper), population_size, selector=selector, offspring_size=offspring_size)
    algorithm.run(WrappedTerminationCondition(evals, tolerance))

    return algorithm


def ga(lower, upper, evals, tolerance, population_size, offspring_size, selector, run=False):
    '''
    1) Initialize a random set of solutions of size population_size
    2) evaluate that population and sort them based on objective to find fittest
    3)
    '''
    if not run:
        return
    algorithm = GeneticAlgorithm(Constraints(lower, upper), population_size, offspring_size, selector=selector)
    algorithm.run(WrappedTerminationCondition(evals, tolerance))

    return algorithm


def pso(lower, upper, evals, tolerance, swarm_size, leader_size, generator, run=False):
    '''
    1) Creates a list of random input sets within the bounds set in problem of length swarm_size called particles
    2) The particles list is evaluated on each iterations after which update_velocities(), update_positions(), mutate()
    3) Update the leaders
    '''
    if not run:
        return
    # https: // deap.readthedocs.io / en / master / api / tools.html  # deap.tools.mutGaussian
    algorithm = ParticleSwarm(Constraints(lower, upper), swarm_size, leader_size, generator,
                              leader_comparator=AttributeDominance(crowding_distance_key),
                              fitness=crowding_distance, fitness_getter=crowding_distance_key)
    # TODO GeneticAlgorithm uses step() method from AbstractGeneticAlgorithm which must be the base class to SingleObjectiveAlgorithm
    #   Line 393 in Core.py sets self.nfe += len(unevaluated)
    #   Line 114 in algorithms.py takes the solutions from each particle and sorts them in order of objective function which sets self.fittest = self.population[0]
    algorithm.run(WrappedTerminationCondition(evals, tolerance))

    return algorithm


def schwefel(x1,x2):
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


def profile_main():

    import cProfile, pstats, io

    lower, upper, num = -500, 500, 100
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)

    prof = cProfile.Profile()
    prof = prof.runctx("nsgaii(lower, upper, run=True)", globals(), locals())

    stream = io.StringIO()

    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("time")  # or cumulative
    stats.print_stats(80)  # 80 = how many to print

    # The rest is optional.
    # stats.print_callees()
    # stats.print_callers()

    # logging.info("Profile data:\n%s", stream.getvalue())

    f = open(os.path.join(OUTPUT_PATH, 'profile.txt'), 'a')
    f.write(stream.getvalue())
    f.close()


def main():
    lower, upper, num = -500, 500, 100
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)
    function_evals = 5000
    tolerance = 10 ** (-4)
    common_params = {'lower': lower, 'upper': upper, 'evals': function_evals, 'tolerance': tolerance}

    # NSGAII
    nsgaii_params = copy.deepcopy(common_params)
    nsgaii_params['population_size'] = 500
    nsgaii_params['offspring_size'] = 1000
    nsgaii_params['selector'] = TournamentSelector(2)
    plotResults('NSGAII', nsgaii(**nsgaii_params, run=False))

    # GA
    ga_params = copy.deepcopy(common_params)
    ga_params['population_size'] = 500
    ga_params['offspring_size'] = 1000
    ga_params['selector'] = TournamentSelector(2)
    plotResults('GA', ga(**ga_params, run=True))

    # PSO
    pso_params = copy.deepcopy(common_params)
    pso_params['generator'] = RandomGenerator()
    pso_params['swarm_size'] = 500
    pso_params['leader_size'] = 1000
    plotResults('PSO', pso(**pso_params, run=False))

    # Plotting
    x1, x2 = np.meshgrid(x1, x2)
    results = schwefel(x1, x2)
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x1, x2, results, cmap=cm.jet, linewidth=0, antialiased=False)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    plt.show()


if __name__ == '__main__':
    # profile_main()
    main()
