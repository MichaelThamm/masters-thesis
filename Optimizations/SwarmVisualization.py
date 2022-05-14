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


class ParticleSolution(object):
    def __init__(self):
        self.particlesPI = {}
        self.velocitiesPI = {}

    def addIterationResults(self, iteration, particles, velocities):
        self.particlesPI[iteration] = copy.deepcopy(particles)
        self.velocitiesPI[iteration] = copy.deepcopy(velocities)


class WrappedTerminationCondition(TerminationCondition):
    def __init__(self, max_evals, tolerance, stall_tolerance, max_stalls, timeout):
        super(TerminationCondition, self).__init__()

        self.log_frequency = 10
        self.store = ParticleSolution()

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

        # Store the particles and velocities
        particles = algorithm.particles if hasattr(algorithm, 'particles') else []
        velocities = algorithm.velocities if hasattr(algorithm, 'velocities') else []
        self.store.addIterationResults(self.iteration, particles, velocities)

        # Determine the fittest based on the algorithm type
        if isinstance(algorithm, AbstractGeneticAlgorithm):
            if len(algorithm.result) > 0:
                fittest = sorted(algorithm.result, key=functools.cmp_to_key(ParetoDominance()))[0]
            else:
                fittest = None
        elif isinstance(algorithm, GeneticAlgorithm):  # GA is a child of AbstractGA but already calculates fittest
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
            logSwarm(self)
            return True

        # Timeout exceeded
        if (time.time()-self.start_time) >= self.timeout:
            self.reason = ExitReason.TIMEOUT
            logTermination(self, fittest)
            logSwarm(self)
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
                logSwarm(self)
                return True
        else:
            self.stall_iteration = 0
            self.old_fittest = fittest.objectives[0]
            return False

        return False


def logTermination(objTermination, fittest):
    message = f'Termination after {objTermination.iteration} iterations due to {objTermination.reason}\n' \
              f'Solution: {fittest.variables} Objective: {fittest.objectives}\n' \
              f'Error: {np.subtract(np.array(fittest.variables), SCHWEFEL_SOLUTION)}'
    logging.getLogger("Platypus").log(logging.INFO, message)


def logSwarm(optimization):
    for iteration, particles in optimization.store.particlesPI.items():
        if iteration % optimization.log_frequency == 0 or iteration == list(optimization.store.particlesPI.keys())[-1]:
            message = f'iteration: {iteration}, particles: {particles}'
            logging.getLogger("Platypus").log(logging.INFO, message)
    for iteration, velocities in optimization.store.velocitiesPI.items():
        if iteration % optimization.log_frequency == 0 or iteration == list(optimization.store.velocitiesPI.keys())[-1]:
            message = f'iteration: {iteration}, velocities: {velocities}'
            logging.getLogger("Platypus").log(logging.INFO, message)


def getSwarm():
    import re

    storeParticles = {}
    storeVelocities = {}
    with open(LOGGER_FILE) as f:
        f = f.readlines()
        for line in f:
            if line.find('iteration: ') != -1:
                try:
                    iteration = re.search('iteration: [0-9]*', line).group(0).split(': ')[1]
                    particles = re.search('particles: .*', line).group(0).split(': ')[1].split(', ')
                except AttributeError:
                    found = ''  # apply your error handling



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


def plottingSchwefel(x1, x2, lower, upper, run=False):
    if not run:
        return

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

    plt.contour(x1, x2, results, 15, colors='grey')
    plt.imshow(results, extent=[lower, upper, lower, upper], origin='lower', cmap=cm.jet, alpha=0.5)
    plt.plot(SCHWEFEL_SOLUTION[0], SCHWEFEL_SOLUTION[1], marker='+', color='red', markersize=12)
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def main():
    # TODO
    #   1) Finish the mutation and selection sections in thesis
    #       a) Run a test case and tabulate it for different tournament sizes
    #   2) Set up the plotting per 10 gens for PSO particles and velocities to be added to the Optimization integration section
    #   2) Integrate some of this code (OMOPSO) into Platypus.py
    #   3) Decide the layout for thesis chapters
    #       *) AT THE END OF EVERY CHAPTER TIE THE MOST IMPORTANT THING BACK TO THE OVERALL OBJECTIVE
    #       a) Chpt1: intro (EV and why is this important, include generics for opt and modeling)
    #           i) At the end of chapt 1 allude to the motor variables to be optimized and what I will be optimizing them for
    #           ii) This can include slot poles and why they produce different performance params. This will be discusses in more detail in Chapt 4 but set it up
    #           iii) talk about limitations of optimal motors like saturation, frequency creating skin effect
    #           *) Anything in here needs to be explained again in later chapters
    #           *) I can include saturation and skin_efect in the constraint of the OMOPSO
    #           *) Talk about localized meshing for efficiency and accuracy
    #       b) Chpt2: Modelling (baseline motor, describe modelling functionality from 2019 paper, include validation at the end of chapter)
    #       c) Chpt3: Optimization (Generic GA and PSO, Case study [talk about PSO only from now on], Selection case study, Variation case study)
    #           i) Include the swarm and GA generational advancement visualization. (arrows for swarm with dot, ga dot)
    #       d) Chpt4: Opt integration (Talk about ratios, constraints)
    #           i) Re-iterate the inputs and outputs, constraints, ... what does an output motor look like?
    #       e) Chpt5: Proposed Design from Opts
    #           i) How did the algorithm perform
    #           ii) Result of the algo
    #           iii) Talk about results (pre-conclusion)
    #       f) Chpt6: Results Discussion and Improvement/Considerations
    #           i) Future work

    # Clear the log file
    logging.FileHandler(LOGGER_FILE, mode='w')

    lower, upper, num = -500, 500, 200
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)
    tolerance = 10 ** (-6)
    max_evals = 30000 - 1

    max_stalls = 25
    stall_tolerance = tolerance
    timeout = 30000  # seconds
    parent_size = 200
    child_size = round(0.25 * parent_size)
    constraint_params = {'lower': lower, 'upper': upper}
    termination_params = {'max_evals': max_evals, 'tolerance': tolerance,
                          'max_stalls': max_stalls, 'stall_tolerance': stall_tolerance,
                          'timeout': timeout}

    pso_params = {'swarm_size': parent_size, 'leader_size': child_size, 'generator': RandomGenerator(), 'mutate': PM(0.1),
                  'leader_comparator': AttributeDominance(crowding_distance_key), 'larger_preferred': True,
                  'fitness': crowding_distance, 'fitness_getter': crowding_distance_key}
    plotResults('PSO', solveOptimization(ParticleSwarm, constraint_params, termination_params, pso_params, run=True))

    getSwarm()

    plottingSchwefel(x1, x2, lower, upper, run=False)


if __name__ == '__main__':
    main()
