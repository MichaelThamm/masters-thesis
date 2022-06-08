import matplotlib.offsetbox as offsetbox
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from time import perf_counter
from itertools import count
from matplotlib import cm
from pathlib import Path
from platypus import *
from enum import Enum
import numpy as np
import logging
import copy
import time
import os

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
https://www.indusmic.com/post/schwefel-function
'''

PROJECT_PATH = Path(__file__).resolve().parent.parent
OPTIMIZATIONS_PATH = os.path.join(PROJECT_PATH, 'Optimizations')
LOGGER_FILE = os.path.join(OPTIMIZATIONS_PATH, 'Solvers.log')

LOGGER = logging.getLogger("Platypus")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.FileHandler(LOGGER_FILE))

SCHWEFEL_SOLUTION = np.array([420.9687, 420.9687])
LOG_EVERY_X_ITERATIONS = 10


class ExitReason(Enum):
    MAX_EVALS = 1
    STALL = 2
    TIMEOUT = 3


class LogHeader(Enum):
    ITERATION = 'iteration'
    BEST = 'best'
    GENERATION = 'generation'


class SchwefelProblem(Problem):

    def __init__(self, lower, upper):
        # The numbers indicate: #inputs, #objectives, #constraints
        super(SchwefelProblem, self).__init__(2, 1)
        # Constrain the range and type for each input
        self.types[:] = [Real(lower, upper), Real(lower, upper)]
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE]

    def evaluate(self, solution):
        x = solution.variables[0]
        y = solution.variables[1]
        magnitude_fitness = schwefel(x, y)
        solution.objectives[:] = [magnitude_fitness]


class GenerationalSolution(object):
    def __init__(self):
        self.solution = {}

    def addIterationResults(self, iteration, solution):
        self.solution[iteration] = copy.deepcopy(solution)


class _SingleSolution:
    def __init__(self, variables, objective):
        self.variables = variables
        self.objective = objective


class _TerminationCondition(TerminationCondition):
    def __init__(self, max_evals, tolerance, stall_tolerance, max_stalls, timeout):
        super(TerminationCondition, self).__init__()

        self.log_frequency = LOG_EVERY_X_ITERATIONS
        self.store = GenerationalSolution()

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

        # Store the solution
        if isinstance(algorithm, ParticleSwarm):
            solution = algorithm.particles if hasattr(algorithm, 'particles') else []
        elif isinstance(algorithm, AbstractGeneticAlgorithm):
            solution = algorithm.population if hasattr(algorithm, 'population') else []
        else:
            solution = None
        self.store.addIterationResults(self.iteration, solution)

        # Determine the fittest based on the algorithm type
        fittest = getFittest(algorithm)

        # First iteration
        if fittest is None:
            return False

        # Max evaluations
        if self.iteration > self.max_evals:
            self.reason = ExitReason.MAX_EVALS
            logTermination(self, fittest)
            logGeneration(self)
            return True

        # Timeout exceeded
        if (time.time()-self.start_time) >= self.timeout:
            self.reason = ExitReason.TIMEOUT
            logTermination(self, fittest)
            logGeneration(self)
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
                logGeneration(self)
                return True
        else:
            self.stall_iteration = 0
            self.old_fittest = fittest.objectives[0]
            return False

        return False


class _RandomIntGenerator(Generator):

    def __init__(self):
        super(Generator, self).__init__()

    def generate(self, problem):
        solution = Solution(problem)
        solution.variables = [x.rand() for x in problem.types]
        return solution


class _ParticlSwarm(ParticleSwarm):
    def __init__(self, problem,
                 swarm_size=100,
                 leader_size=100,
                 generator=RandomGenerator(),
                 mutate=None,
                 leader_comparator=AttributeDominance(fitness_key),
                 dominance=ParetoDominance(),
                 fitness=None,
                 larger_preferred=True,
                 fitness_getter=fitness_key,
                 **kwargs):
        super().__init__(problem,
                         swarm_size=swarm_size,
                         leader_size=leader_size,
                         generator=generator,
                         mutate=mutate,
                         leader_comparator=leader_comparator,
                         dominance=dominance,
                         fitness=fitness,
                         larger_preferred=larger_preferred,
                         fitness_getter=fitness_getter,
                         **kwargs)

    def iterate(self):
        self._update_velocities()
        self._update_positions()
        self._mutate()
        self.evaluate_all(self.particles)
        self._update_local_best()

        self.leaders += self.particles
        self.leaders.truncate(self.leader_size)

    def _mutate(self):
        if self.mutate is not None:
            for i in range(self.swarm_size):
                self.particles[i] = self.mutate.mutate(self.particles[i])


def logTermination(objTermination, fittest):
    message = f'Termination after {objTermination.iteration} iterations due to {objTermination.reason}\n' \
              f'Solution: {fittest.variables} Objective: {fittest.objectives}\n' \
              f'Error: {np.subtract(np.array(fittest.variables), SCHWEFEL_SOLUTION)}'
    logging.getLogger("Platypus").log(logging.INFO, message)


def logGeneration(optimization):
    for iteration, generation in optimization.store.solution.items():
        if iteration % optimization.log_frequency == 0 or iteration in [2, list(optimization.store.solution.keys())[-1]]:
            message = f'{LogHeader.ITERATION.value}: {iteration}, ' \
                      f'{LogHeader.BEST.value}: {sorted(generation, key=functools.cmp_to_key(ParetoDominance()))[0]}, ' \
                      f'{LogHeader.GENERATION.value}: {generation}'
            logging.getLogger("Platypus").log(logging.INFO, message)


def getSolutionFromLog(log_order):
    import re

    def strToFloat(str_in):
        return eval(f'float(str_in)')

    def extractStrSolution(str_in):
        variables = str_in.split('[')
        nums = variables[1].split(',')
        num1 = nums[0]
        temp = nums[1].split('|')
        num2 = temp[0]
        objective = temp[1]
        return _SingleSolution([strToFloat(num1), strToFloat(num2)], strToFloat(objective))

    def time_string_to_decimals(time_string):
        fields = time_string.split(":")
        hours = fields[0] if len(fields) > 0 else 0.0
        minutes = fields[1] if len(fields) > 1 else 0.0
        seconds = fields[2] if len(fields) > 2 else 0.0
        return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)

    # Ensure that there are no duplicate solution names in the solverList
    archive = copy.deepcopy(log_order)
    for cnt, log_name in enumerate(archive):
        zippedCount = [(i, j) for i, j in zip(count(), log_order) if j == log_name]
        if len(zippedCount) > 1:
            for index, name in zippedCount:
                log_order[index] += f'{index}'

    sectionIdxs = {}
    storeSolution, storeSolutions = {}, {}
    with open(LOGGER_FILE) as f:
        f = f.readlines()

        # Define the indexes for each solution section in the log
        sectionFooters = list(map(lambda x: x.find('finished; Total NFE:') != -1, f))
        sectionFooterIdxs = [0] + [index for index, value in enumerate(sectionFooters) if value]
        for cnt, idx in enumerate(sectionFooterIdxs):
            if cnt < len(sectionFooterIdxs) - 1:
                sectionIdxs[log_order[cnt]] = list(range(sectionFooterIdxs[cnt], sectionFooterIdxs[cnt + 1] + 1))

        # Store lines in the log that contain population information
        for section, indexes in sectionIdxs.items():
            for index in indexes:
                line = f[index]
                if line.find(f'{LogHeader.ITERATION.value}: ') != -1:
                    try:
                        iteration = int(re.search(f'{LogHeader.ITERATION.value}: ([0-9]*)', line).group(1))
                        best = re.search(f'{LogHeader.BEST.value}: (.*?), {LogHeader.GENERATION.value}', line).group(1)
                        solutions = re.search(f'{LogHeader.GENERATION.value}: (.*)', line).group(1)[1:-1].split(', ')

                        # Store solutions
                        storeWrappedSolutions = {}
                        for cnt, solution in enumerate(solutions):
                            storeWrappedSolutions[cnt] = extractStrSolution(solution)
                        storeSolution[iteration] = {LogHeader.BEST.value: extractStrSolution(best),
                                                    LogHeader.GENERATION.value: storeWrappedSolutions}

                    except AttributeError:
                        pass

                    except IndexError:
                        pass
                elif line.find('finished; Total NFE: ') != -1:
                    try:
                        nfe = int(re.search(f'finished; Total NFE: ([0-9]*)', line).group(1))
                        _time = time_string_to_decimals(re.search(f'Elapsed Time: (.*)', line).group(1))

                    except AttributeError:
                        pass

                    except IndexError:
                        pass
            storeSolutions[section] = {'summary': {'nfe': nfe, 'time': _time}, 'generation': storeSolution}
            storeSolution = {}

    return storeSolutions


def getFittest(algorithm):
    if isinstance(algorithm, GeneticAlgorithm):  # GA is a child of AbstractGA but already calculates fittest
        if hasattr(algorithm, 'fittest'):
            fittest = algorithm.fittest
        else:
            fittest = None
    elif isinstance(algorithm, AbstractGeneticAlgorithm):
        if len(algorithm.result) > 0:
            fittest = sorted(algorithm.population, key=functools.cmp_to_key(ParetoDominance()))[0]
        else:
            fittest = None
    elif isinstance(algorithm, ParticleSwarm):
        if hasattr(algorithm, 'leaders'):
            fittest = algorithm.leaders._contents[0]  # Use this for efficiency
        else:
            fittest = None
    else:
        fittest = None

    return fittest


def solveOptimization(algorithm, problem, solverList, constraint, termination, solver, run=False):
    if not run:
        return solverList
    optimization = algorithm(problem(**constraint), **solver)
    optimization.run(_TerminationCondition(**termination))
    if hasattr(algorithm, '__name__'):
        return addAlgoName(solverList, algorithm.__name__)
    else:
        return addAlgoName(solverList, type(algorithm).__name__)


def schwefel(x1, x2):
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


def plottingSchwefel(x1, x2, lower, upper, run=False):
    if not run:
        return

    # Plotting
    x1, x2 = np.meshgrid(x1, x2)
    space = schwefel(x1, x2)
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x1, x2, space, rstride=1, cstride=1, cmap=cm.get_cmap('jet'), linewidth=0, antialiased=False)
    axis.set_xlabel('x1')
    axis.set_ylabel('x2')
    axis.set_zlabel('f(x1,x2)')
    plt.show()

    plt.contour(x1, x2, space, 15, colors='grey')
    plt.imshow(space, extent=[lower, upper, lower, upper], origin='lower', cmap=cm.get_cmap('jet'), alpha=0.5)
    plt.plot(SCHWEFEL_SOLUTION[0], SCHWEFEL_SOLUTION[1], marker='+', color='red', markersize=12)
    plt.colorbar()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def plottingConvergence(x1, x2, lower, upper, solutions, run=False):
    if not run:
        return

    x1, x2 = np.meshgrid(x1, x2)
    space = schwefel(x1, x2)
    for algorithmName, solverSolution in solutions.items():
        for solutionName, solutionType in solverSolution.items():
            if solutionName == LogHeader.GENERATION.value:
                for iteration, solution in solutionType.items():
                    plt.contour(x1, x2, space, 15, colors='grey', zorder=-1)
                    plt.imshow(space, extent=[lower, upper, lower, upper], origin='lower', cmap=cm.get_cmap('jet'), alpha=0.5)
                    plt.colorbar()
                    xVariables = [value.variables[0] for value in solution[LogHeader.GENERATION.value].values()]
                    yVariables = [value.variables[1] for value in solution[LogHeader.GENERATION.value].values()]
                    plt.scatter(xVariables, yVariables, marker='*', color='green')
                    plt.scatter(solution[LogHeader.BEST.value].variables[0], solution[LogHeader.BEST.value].variables[1],
                                marker='*', color='red')
                    plt.plot(SCHWEFEL_SOLUTION[0], SCHWEFEL_SOLUTION[1], marker='+', color='red', markersize=12)
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    formattedObj = "{:.4f}".format(solution[LogHeader.BEST.value].objective)
                    formattedVars = ["{:.4f}".format(x) for x in solution[LogHeader.BEST.value].variables]
                    plt.title(f'{algorithmName} Iteration: {iteration}\nBest: {formattedObj}@{formattedVars}')
                    plt.show()


def plottingPerformance(solvers, data, plot=False):

    plotDict = {}
    for name in solvers:
        plotDict[name] = {'iterations': np.array([iteration for iteration in data[name]['generation']]),
                          'best_objectives': np.array([generation[LogHeader.BEST.value].objective for iteration, generation in data[name]['generation'].items()]),
                          'best_variables': np.array([generation[LogHeader.BEST.value].variables for iteration, generation in data[name]['generation'].items()]),
                          'nfe': data[name]['summary']['nfe'],
                          'time': data[name]['summary']['time']}

        y_normal = plotDict[name]['best_objectives']
        y_logarithmic = np.log(y_normal)

        if plot:
            fig, ax1 = plt.subplots()
            ax1.title.set_text(f"{name}\nObjective Value (Logarithmic-scale) vs Iterations")
            stats = f"Best: {'{:.4f}'.format(min(y_normal))}\n" \
                    f"Function Executions: {plotDict[name]['nfe']}\n" \
                    f"Time: {'{:.4f}'.format(plotDict[name]['time'])}"

            ob = offsetbox.AnchoredText(stats, loc='upper right', prop=dict(color='black', size=10))
            ob.patch.set(boxstyle='round', color='grey', alpha=0.5)
            ax1.add_artist(ob)

            ax1.scatter(plotDict[name]['iterations'], y_logarithmic)
            ax2 = ax1.twinx()
            ax2.set_ylim([min(y_normal), max(y_normal)])
            ax1.set_xlabel('Iterations')
            ax1.set_ylabel('Logarithmic Scale of Objective Value')
            ax2.set_ylabel('Objective Value')
            plt.show()

    return plotDict


def addAlgoName(solverList, name):
    if name != 'NoneType':
        solverList.append(name)
    return solverList


def main():

    # TODO
    #   Decide the layout for thesis chapters
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

    lower, upper, num = -500, 500, 100
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)
    tolerance = 10 ** (-7)
    max_evals = 30000 - 1

    max_stalls = 25
    stall_tolerance = tolerance
    timeout = 3000000  # seconds
    parent_size = 200
    offspring_size = round(0.5 * parent_size)
    leader_size = round(1.0 * parent_size)
    # tournament_size = round(0.25 * parent_size)
    tournament_size = 2
    constraint_params = {'lower': lower, 'upper': upper}
    termination_params = {'max_evals': max_evals, 'tolerance': tolerance,
                          'max_stalls': max_stalls, 'stall_tolerance': stall_tolerance,
                          'timeout': timeout}
    solverList = []

    ga_params = {'population_size': parent_size, 'offspring_size': offspring_size, 'generator': RandomGenerator(),
                 'selector': TournamentSelector(tournament_size), 'comparator': ParetoDominance(),
                 'variator': GAOperator(SBX(0.3), PM(0.1))}
    solverList = solveOptimization(GeneticAlgorithm, SchwefelProblem, solverList, constraint_params, termination_params, ga_params, run=True)

    # TODO NSGA doesnt have offspring parameter because it clips to population size instead
    #  -> if pop = 200 then offspring + pop = 400 which is fitness sorted and clipped to 200 again
    nsga_params = {'population_size': parent_size, 'generator': RandomGenerator(),
                   'selector': TournamentSelector(tournament_size), 'variator': GAOperator(SBX(0.3), PM(0.1)),
                   'archive': FitnessArchive(nondominated_sort)}
    solverList = solveOptimization(NSGAII, SchwefelProblem, solverList, constraint_params, termination_params, nsga_params, run=True)

    pso_params = {'swarm_size': parent_size, 'leader_size': leader_size, 'generator': RandomGenerator(),
                  'mutate': PM(0.1), 'leader_comparator': AttributeDominance(objective_key),
                  'larger_preferred': True, 'fitness': crowding_distance, 'fitness_getter': objective_key}
    solverList = solveOptimization(_ParticlSwarm, SchwefelProblem, solverList, constraint_params, termination_params, pso_params, run=True)

    omopso_params = {'epsilons': [0.05], 'swarm_size': parent_size, 'leader_size': leader_size,
                     'mutation_probability': 0.1, 'mutation_perturbation': 0.5, 'max_iterations': 100,
                     'generator': RandomGenerator(), 'selector': TournamentSelector(tournament_size), 'variator': SBX(0.3)}
    solverList = solveOptimization(OMOPSO, SchwefelProblem, solverList, constraint_params, termination_params, omopso_params, run=True)

    solutions = getSolutionFromLog(solverList)

    summarizedResult = plottingPerformance(solverList, solutions, plot=True)
    for name, generations in summarizedResult.items():
        print(f"________{name}________\n"
              f"objective: {generations['best_objectives'][-1]}\n"
              f"variables: {generations['best_variables'][-1]}\n"
              f"nfe: {generations['nfe']}\n"
              f"time: {generations['time']}\n")
    plottingConvergence(x1, x2, lower, upper, solutions, run=True)
    plottingSchwefel(x1, x2, lower, upper, run=True)


if __name__ == '__main__':
    main()
