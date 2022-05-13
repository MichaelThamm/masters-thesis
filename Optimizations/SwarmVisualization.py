from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from platypus import *
from enum import Enum
import numpy as np
import logging
import os

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
https://www.indusmic.com/post/schwefel-function
'''

PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
OPTIMIZATIONS_PATH = os.path.join(PROJECT_PATH, 'Optimizations')
LOGGER_FILE = os.path.join(OPTIMIZATIONS_PATH, 'Swarm.log')

LOGGER = logging.getLogger("Platypus")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.FileHandler(LOGGER_FILE))

SCHWEFEL_SOLUTION = np.array([420.9687, 420.9687])


class ExitReason(Enum):
    MAX_EVALS = 1
    STALL = 2
    TIMEOUT = 3


class Constraints(Problem):

    def __init__(self, lower, upper):
        # The numbers indicate: #inputs, #objectives, #constraints
        super(Constraints, self).__init__(2, 1)
        # Constrain the range and type for each input
        self.types[:] = [Real(lower, upper), Real(lower, upper)]
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE]

    def evaluate(self, solution):
        x = solution.variables[0]
        y = solution.variables[1]
        magnitude_fitness = schwefel(x, y)
        solution.objectives[:] = [magnitude_fitness]


class WrappedTerminationCondition(TerminationCondition):
    def __init__(self, max_evals, tolerance, stall_tolerance, max_stalls, timeout):
        super(TerminationCondition, self).__init__()
        self.iteration = 0
        self.stall_iteration = 0
        self.max_evals = max_evals
        self.tolerance = tolerance
        self.stall_tolerance = stall_tolerance
        self.max_stalls = max_stalls
        self.start_time = time.time()
        self.timeout = timeout
        self.old_fittest = np.inf
        self.reason = None

    def __call__(self, algorithm):
        return self.shouldTerminate(algorithm)

    def initialize(self, algorithm):
        pass

    def shouldTerminate(self, algorithm):
        self.iteration += 1
        if isinstance(algorithm, AbstractGeneticAlgorithm):
            if len(algorithm.result) > 0:
                fittest = sorted(algorithm.result, key=functools.cmp_to_key(ParetoDominance()))[0]
            else:
                fittest = None
        elif isinstance(algorithm, GeneticAlgorithm):
            if hasattr(algorithm, 'fittest'):
                fittest = algorithm.fittest
            else:
                fittest = None
        elif isinstance(algorithm, ParticleSwarm):
            if hasattr(algorithm, 'leaders'):
                fittest = algorithm.leaders._contents[0]  # Use this for efficiency
            else:
                fittest = None
        else:
            fittest = None

        # First iteration
        if fittest is None:
            return False

        # Max evaluations
        if self.iteration > self.max_evals:
            self.reason = ExitReason.MAX_EVALS
            logTermination(self, fittest)
            return True

        # Timeout exceeded
        if (time.time()-self.start_time) >= self.timeout:
            self.reason = ExitReason.TIMEOUT
            logTermination(self, fittest)
            return True

        # Objective tolerance achieved
        if fittest.objectives[0] <= self.tolerance:
            return True

        # Objective plateau
        if self.old_fittest - fittest.objectives[0] <= self.stall_tolerance:
            self.stall_iteration += 1
            if self.stall_iteration >= self.max_stalls:
                self.reason = ExitReason.STALL
                logTermination(self, fittest)
                return True
        else:
            self.stall_iteration = 0
            self.old_fittest = fittest.objectives[0]
            return False

        return False


def logTermination(objTermination, fittest):
    message = f'Termination after {objTermination.iteration - 1} iterations due to {objTermination.reason}\n' \
              f'Solution: {fittest.variables} Objective: {fittest.objectives}\n' \
              f'Error: {np.subtract(np.array(fittest.variables), SCHWEFEL_SOLUTION)}'
    logging.getLogger("Platypus").log(logging.INFO, message)


def plotResults(str_name, algorithm):
    if not algorithm:
        return
    feasible_solutions = [s for s in unique(nondominated(algorithm.result))]
    plotResults = []
    for results in feasible_solutions:
        x, y = (algorithm.problem.types[0].decode(results.variables[0]),
                algorithm.problem.types[0].decode(results.variables[1]))
        objective = results.objectives[0]
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


def solveOptimization(algorithm, constraint, termination, solver, run=False):
    if not run:
        return
    optimization = algorithm(Constraints(**constraint), **solver)
    optimization.run(WrappedTerminationCondition(**termination))

    return optimization


def schwefel(x1, x2):
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


def main():
    # Clear the log file
    logging.FileHandler(LOGGER_FILE, mode='w')

    lower, upper, num = -500, 500, 1000
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)
    tolerance = 10 ** (-6)
    max_evals = 30000
    max_stalls = 25
    stall_tolerance = tolerance
    timeout = 30000  # seconds
    parent_size = 1000
    child_size = round(0.25 * parent_size)
    constraint_params = {'lower': lower, 'upper': upper}
    termination_params = {'max_evals': max_evals, 'tolerance': tolerance,
                          'max_stalls': max_stalls, 'stall_tolerance': stall_tolerance,
                          'timeout': timeout}

    pso_params = {'swarm_size': parent_size, 'leader_size': child_size, 'generator': RandomGenerator(), 'mutate': None,
                  'leader_comparator': AttributeDominance(crowding_distance_key), 'larger_preferred': True,
                  'fitness': crowding_distance, 'fitness_getter': crowding_distance_key}
    plotResults('PSO', solveOptimization(ParticleSwarm, constraint_params, termination_params, pso_params, run=True))

    x1, x2 = np.meshgrid(x1, x2)
    results = schwefel(x1, x2)
    plt.contour(x1, x2, results, 15, colors='grey')
    plt.imshow(results, extent=[lower, upper, lower, upper], origin='lower', cmap=cm.jet, alpha=0.5)
    plt.plot(SCHWEFEL_SOLUTION[0], SCHWEFEL_SOLUTION[1], marker='+', color='red', markersize=12)
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


if __name__ == '__main__':
    main()
