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
LOGGER_FILE = os.path.join(OPTIMIZATIONS_PATH, 'Optimization.log')

LOGGER = logging.getLogger("Platypus")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.FileHandler(LOGGER_FILE))

SCHWEFEL_SOLUTION = np.array([420.9687, 420.9687])


class ExitReason(Enum):
    MAX_EVALS = 1
    STALL = 2
    TIMEOUT = 3


# Problem is the object that Test will inherit from
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
    def __init__(self, max_evals, tolerance, max_stalls, timeout):
        super(TerminationCondition, self).__init__()
        self.iteration = 0
        self.stall_iteration = 0
        self.max_evals = max_evals
        self.solver_tolerance = tolerance
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
        # if isinstance(algorithm, AbstractGeneticAlgorithm):
        #     if len(algorithm.result) > 0:
        #         fittest = sorted(algorithm.result, key=functools.cmp_to_key(ParetoDominance()))[0]
        #     else:
        #         fittest = None
        # elif isinstance(algorithm, GeneticAlgorithm):
        #     if hasattr(algorithm, 'fittest'):
        #         fittest = algorithm.fittest
        #     else:
        #         fittest = None
        # elif isinstance(algorithm, ParticleSwarm):
        #     if hasattr(algorithm, 'leaders'):
        #         # self.fitness(self._contents)
        #         fittest = algorithm.leaders._contents[0]  # Use this for efficiency
        #     else:
        #         fittest = None
        # else:
        #     fittest = None

        # Max evaluations
        if self.iteration > self.max_evals:
            self.reason = ExitReason.MAX_EVALS

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

            LOGGER.log(logging.INFO, f'Termination after {self.iteration - 1} iterations due to {self.reason}\n'
                                     f'Solution: {fittest.variables} Objective: {fittest.objectives}\n'
                                     f'Error: {np.subtract(np.array(fittest.variables), SCHWEFEL_SOLUTION)}')

            return True
        else:
            return False
        # # Timeout exceeded
        # if (time.time()-self.start_time) >= self.timeout:
        #     self.reason = ExitReason.TIMEOUT
        #     LOGGER.log(logging.INFO, f'Termination after {self.iteration - 1} iterations due to {self.reason}\n'
        #                              f'Solution: {fittest.variables} Objective: {fittest.objectives}\n'
        #                              f'Error: {np.subtract(np.array(fittest.variables), SCHWEFEL_SOLUTION)}')
        #     return True
        # # Objective plateau
        # if fittest is None:
        #     return False
        # else:
        #     if self.old_fittest - fittest.objectives[0] <= self.solver_tolerance:
        #         self.stall_iteration += 1
        #         if self.stall_iteration >= self.max_stalls:
        #             self.reason = ExitReason.STALL
        #             LOGGER.log(logging.INFO, f'Termination after {self.iteration - 1} iterations due to {self.reason}\n'
        #                                      f'Solution: {fittest.variables} Objective: {fittest.objectives}\n'
        #                                      f'Error: {np.subtract(np.array(fittest.variables), SCHWEFEL_SOLUTION)}')
        #             return True
        #     else:
        #         self.stall_iteration = 0
        #         self.old_fittest = fittest.objectives[0]
        #         return False


def plotResults(str_name, algorithm):
    if not algorithm:
        return
    feasible_solutions = [s for s in unique(nondominated(algorithm.result))]
    plotResults = []
    for results in feasible_solutions:
        x, y = (algorithm.problem.types[0].decode(results.variables[0]), algorithm.problem.types[0].decode(results.variables[1]))
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


def schwefel(x1,x2):
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


def main():

    # Clear the log file
    logging.FileHandler(LOGGER_FILE, mode='w')

    lower, upper, num = -500, 500, 100
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)
    max_evals = 30
    max_stalls = 10
    tolerance = 10 ** (-4)
    timeout = 30  # seconds
    parent_size = 1000
    child_size = round(0.25 * parent_size)
    tournament_size = round(0.1 * parent_size)
    constraint_params = {'lower': lower, 'upper': upper}
    termination_params = {'max_evals': max_evals, 'tolerance': tolerance, 'max_stalls': max_stalls,
                          'timeout': timeout}

    # TODO from .config import default_variator, default_mutator are the default mutation or selection algorithms

    # NSGAII
    nsgaii_params = {'population_size': parent_size, 'offspring_size': child_size, 'generator': RandomGenerator(),
                     'selector': TournamentSelector(tournament_size), 'archive': None, 'variator': None}
    plotResults('NSGAII', solveOptimization(NSGAII, constraint_params, termination_params, nsgaii_params, run=False))

    # GA
    '''
    1) Initialize a random set of solutions of size population_size
    2) evaluate that population and sort them based on objective to find fittest
    3)
    '''
    ga_params = {'population_size': parent_size, 'offspring_size': child_size, 'generator': RandomGenerator(),
                 'selector': TournamentSelector(tournament_size), 'variator': None}
    plotResults('GA', solveOptimization(GeneticAlgorithm, constraint_params, termination_params, ga_params, run=False))

    # PSO
    '''
    1) Creates a list of random input sets within the bounds set in problem of length swarm_size called particles
    2) The particles list is evaluated on each iterations after which update_velocities(), update_positions(), mutate()
    3) Update the leaders
    '''

    pso_params = {'swarm_size': parent_size, 'leader_size': child_size, 'generator': RandomGenerator(), 'mutate': None,
                  'leader_comparator': AttributeDominance(crowding_distance_key), 'larger_preferred': True,
                  'fitness': crowding_distance, 'fitness_getter': crowding_distance_key}
    plotResults('PSO', solveOptimization(ParticleSwarm, constraint_params, termination_params, pso_params, run=False))

    # OMOPSO
    omopso_params = {'epsilons': [0.05], 'swarm_size': parent_size, 'leader_size': child_size,
                     'mutation_probability': 0.1, 'mutation_perturbation': 0.5, 'max_iterations': 100,
                     'generator': RandomGenerator(), 'selector': TournamentSelector(tournament_size), 'variator': None}
    plotResults('OMOPSO', solveOptimization(OMOPSO, constraint_params, termination_params, omopso_params, run=False))

    # Plotting
    x1, x2 = np.meshgrid(x1, x2)
    results = schwefel(x1, x2)
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x1, x2, results, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    axis.set_xlabel('x1')
    axis.set_ylabel('x2')
    axis.set_zlabel('f(x1,x2)')
    plt.show()


if __name__ == '__main__':
    main()
