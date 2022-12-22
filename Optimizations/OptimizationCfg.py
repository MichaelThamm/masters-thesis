from LIM.Compute import *
import LIM.Show

import matplotlib.offsetbox as offsetbox
import matplotlib.pyplot as plt
from itertools import count
from matplotlib import cm
from pathlib import Path
from platypus import *
from enum import Enum
import numpy as np
import logging
import json
import math
import copy
import time
import os

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
https://www.indusmic.com/post/schwefel-function
'''

PROJECT_PATH = Path(__file__).resolve().parent.parent

OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Output')
DATA_PATH = os.path.join(OUTPUT_PATH, 'StoredSolutionData.json')
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


class AbstractProblem(Problem):
    def __init__(self, **kwargs):
        super(AbstractProblem, self).__init__(**kwargs)

    @abstractmethod
    def evaluate(self, solution):
        raise NotImplementedError("method not implemented")

    @abstractmethod
    def logTermination(self, objTermination, fittest):
        raise NotImplementedError("method not implemented")

    @staticmethod
    def getSolution(algorithm):
        PARTICLES = 'particles'
        POPULATION = 'population'
        if isinstance(algorithm, ParticleSwarm) and hasattr(algorithm, PARTICLES):
            return algorithm.__dict__[PARTICLES]
        elif isinstance(algorithm, AbstractGeneticAlgorithm) and hasattr(algorithm, POPULATION):
            return algorithm.__dict__[POPULATION]
        # First iteration or algorithm is not implemented
        else:
            return []

    @staticmethod
    def decodeSolution(algorithm, fittest):
        localFittest = copy.deepcopy(fittest)
        if all([isinstance(localFittest.variables[i], list) for i in range(algorithm.problem.__dict__["nvars"])]):
            for i in range(algorithm.problem.__dict__["nvars"]):
                localFittest.__dict__["variables"][i] = algorithm.problem.types[i].decode(
                    localFittest.__dict__["variables"][i])
        return localFittest

    @staticmethod
    def logGeneration(optimization):
        for iteration, generation in optimization.store.solution.items():
            if iteration % optimization.log_frequency == 0 or iteration in [2,
                                                                            list(optimization.store.solution.keys())[
                                                                                -1]]:
                message = f'{LogHeader.ITERATION.value}: {iteration}, ' \
                          f'{LogHeader.BEST.value}: {sorted(generation, key=functools.cmp_to_key(ParetoDominance()))[0]}, ' \
                          f'{LogHeader.GENERATION.value}: {generation}'
                LOGGER.log(logging.INFO, message)

    @staticmethod
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
                    sectionIdxs[log_order[cnt]] = list(
                        range(sectionFooterIdxs[cnt], sectionFooterIdxs[cnt + 1] + 1))

            # Store lines in the log that contain population information
            for section, indexes in sectionIdxs.items():
                for index in indexes:
                    line = f[index]
                    if line.find(f'{LogHeader.ITERATION.value}: ') != -1:
                        try:
                            iteration = int(re.search(f'{LogHeader.ITERATION.value}: ([0-9]*)', line).group(1))
                            best = re.search(f'{LogHeader.BEST.value}: (.*?), {LogHeader.GENERATION.value}',
                                             line).group(1)
                            solutions = re.search(f'{LogHeader.GENERATION.value}: (.*)', line).group(1)[1:-1].split(
                                ', ')

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

    @staticmethod
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


class MotorOptProblem(AbstractProblem):
    def __init__(self, slots, pole_pairs, motorCfg, hamCfg, canvasCfg):
        # The numbers indicate: #inputs, #objectives, #constraints
        super(MotorOptProblem, self).__init__(nvars=2, nobjs=2, nconstrs=1)
        # Constrain the range and type for each input
        self.types[:] = [Integer(slots[0], slots[1]), Integer(pole_pairs[0], pole_pairs[1])]
        # Constrain every input. This works in unison with with constraints in the evaluate method
        self.constraints[:] = "==0"
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE, self.MAXIMIZE]
        self.motorCfg = motorCfg
        self.hamCfg = hamCfg
        self.canvasCfg = canvasCfg
        self.validMotors = set([])

    def evaluate(self, solution):
        slots = solution.variables[0]
        polePairs = solution.variables[1]

        self.motorCfg['slots'] = slots
        self.motorCfg['pole_pairs'] = polePairs

        print("outside", slots, polePairs)
        if self.isFeasibleMotor(m=3) == 0:
            # Object for the model design, grid, and matrices
            model = buildMotor(run=True, baseline=False, optimize=True,
                               motorCfg=self.motorCfg, hamCfg=self.hamCfg, canvasCfg=self.canvasCfg)
            mass_fitness = model.massTot
            thrust_fitness = model.Fx.real
            self.validMotors.add((model.slots, model.polePairs))
        else:
            mass_fitness = math.inf
            thrust_fitness = 0

        # print('solution: ', slots, polePairs, mass_fitness, thrust_fitness)

        solution.objectives[:] = [mass_fitness, thrust_fitness]
        solution.constraints[:] = [self.isFeasibleMotor(m=3)]

    def logTermination(self, objTermination, fittest):
        mReason = f'Termination after {objTermination.iteration} iterations due to {objTermination.reason}\n'
        mSolution = f'Solution: {fittest.variables} Objective: {fittest.objectives}\n'
        validMotors = objTermination.problem.validMotors
        mValidMotors = f'Valid motors len: {len(validMotors)} - {validMotors}\n'
        mAllMotors = f'All motors len: {len(self.uniqueCombinations())} - {self.uniqueCombinations()}\n'
        message = mReason + mSolution + mValidMotors + mAllMotors
        LOGGER.log(logging.INFO, message)

    def isFeasibleMotor(self, m):
        slots, polePairs = self.motorCfg['slots'], self.motorCfg['pole_pairs']
        q = slots / (2 * polePairs * m)
        if slots > (2 * polePairs) and slots % m == 0 and q % 1 == 0 and polePairs >= 2:
            return 0
        else:
            return 1

    def uniqueCombinations(self):
        slotsList = range(self.types[0].min_value, self.types[0].max_value + 1)
        polesPairsList = range(self.types[1].min_value, self.types[1].max_value + 1)
        result = []
        for slots, polePairs in itertools.product(slotsList, polesPairsList):
            self.motorCfg["slots"] = slots
            self.motorCfg['pole_pairs'] = polePairs
            if self.isFeasibleMotor(m=3) == 0:
                result.append((slots, polePairs))
        return result


class SchwefelProblem(AbstractProblem):

    def __init__(self, lower, upper):
        # The numbers indicate: #inputs, #objectives, #constraints
        super(SchwefelProblem, self).__init__(nvars=2, nobjs=1)
        # Constrain the range and type for each input
        self.types[:] = [Real(lower, upper), Real(lower, upper)]
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE]

    def evaluate(self, solution):
        x = solution.variables[0]
        y = solution.variables[1]
        magnitude_fitness = self.schwefel(x, y)
        solution.objectives[:] = [magnitude_fitness]

    def logTermination(self, objTermination, fittest):
        mReason = f'Termination after {objTermination.iteration} iterations due to {objTermination.reason}\n'
        mSolution = f'Solution: {fittest.variables} Objective: {fittest.objectives}\n'
        mError = f'Error: {np.subtract(np.array(fittest.variables), SCHWEFEL_SOLUTION)}'
        message = mReason + mSolution + mError
        LOGGER.log(logging.INFO, message)

    @staticmethod
    def schwefel(x1, x2):
        dims = 2
        return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))

    @staticmethod
    def plottingSchwefel(x1, x2, lower, upper, run=False):
        if not run:
            return

        # Plotting
        x1, x2 = np.meshgrid(x1, x2)
        space = SchwefelProblem.schwefel(x1, x2)
        fig = plt.figure()
        axis = fig.add_subplot(projection='3d')
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

    @staticmethod
    def plottingConvergence(x1, x2, lower, upper, solutions, run=False):
        if not run:
            return

        x1, x2 = np.meshgrid(x1, x2)
        space = SchwefelProblem.schwefel(x1, x2)
        for algorithmName, solverSolution in solutions.items():
            for solutionName, solutionType in solverSolution.items():
                if solutionName == LogHeader.GENERATION.value:
                    for iteration, solution in solutionType.items():
                        plt.contour(x1, x2, space, 15, colors='grey', zorder=-1)
                        plt.imshow(space, extent=[lower, upper, lower, upper], origin='lower', cmap=cm.get_cmap('jet'),
                                   alpha=0.5)
                        plt.colorbar()
                        xVariables = [value.variables[0] for value in solution[LogHeader.GENERATION.value].values()]
                        yVariables = [value.variables[1] for value in solution[LogHeader.GENERATION.value].values()]
                        plt.scatter(xVariables, yVariables, marker='*', color='green')
                        plt.scatter(solution[LogHeader.BEST.value].variables[0],
                                    solution[LogHeader.BEST.value].variables[1],
                                    marker='*', color='red')
                        plt.plot(SCHWEFEL_SOLUTION[0], SCHWEFEL_SOLUTION[1], marker='+', color='red', markersize=12)
                        plt.xlabel('x1')
                        plt.ylabel('x2')
                        formattedObj = "{:.4f}".format(solution[LogHeader.BEST.value].objective)
                        formattedVars = ["{:.4f}".format(x) for x in solution[LogHeader.BEST.value].variables]
                        plt.title(f'{algorithmName} Iteration: {iteration}\nBest: {formattedObj}@{formattedVars}')
                        plt.show()

    @staticmethod
    def plottingPerformance(solvers, data, plot=False):
        plotDict = {}
        for name in solvers:
            plotDict[name] = {'iterations': np.array([iteration for iteration in data[name]['generation']]),
                              'best_objectives': np.array(
                                  [generation[LogHeader.BEST.value].objective for iteration, generation in
                                   data[name]['generation'].items()]),
                              'best_variables': np.array(
                                  [generation[LogHeader.BEST.value].variables for iteration, generation in
                                   data[name]['generation'].items()]),
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
    def __init__(self, problem, max_evals, tolerance, stall_tolerance, max_stalls, timeout):
        super(_TerminationCondition, self).__init__()

        self.log_frequency = LOG_EVERY_X_ITERATIONS
        self.store = GenerationalSolution()
        self.problem = problem

        self.iteration = 0
        self.stall_iteration = 0
        self.max_evals = max_evals
        self.tolerance = tolerance
        self.stall_tolerance = stall_tolerance
        self.max_stalls = max_stalls
        self.start_time = time.time()
        self.timeout = timeout
        self.old_fittest = self.initFitness()
        self.reason = None

    def initialize(self, algorithm):
        pass

    def shouldTerminate(self, algorithm):
        self.iteration += 1

        solution = self.problem.getSolution(algorithm)

        if len(solution) != 0:
            self.store.addIterationResults(self.iteration, solution)
            # Determine the fittest based on the algorithm type
            fittest = self.problem.decodeSolution(algorithm, self.problem.getFittest(algorithm))
        # First iteration
        else:
            return False

        # Max evaluations
        if self.iteration > self.max_evals:
            self.reason = ExitReason.MAX_EVALS
            self.problem.logTermination(self, fittest)
            self.problem.logGeneration(self)
            return True

        # Timeout exceeded
        if (time.time() - self.start_time) >= self.timeout:
            self.reason = ExitReason.TIMEOUT
            self.problem.logTermination(self, fittest)
            self.problem.logGeneration(self)
            return True

        # Objective tolerance achieved
        if fittest.objectives[0] <= self.tolerance:
            return True

        # Objective plateau
        isSame = all([fittest.objectives[cnt] - each <= self.stall_tolerance for cnt, each in enumerate(self.old_fittest)])
        if isSame:
            self.stall_iteration += 1
            if self.stall_iteration >= self.max_stalls:
                self.reason = ExitReason.STALL
                self.problem.logTermination(self, fittest)
                self.problem.logGeneration(self)
                return True
        else:
            self.stall_iteration = 0
            self.old_fittest = fittest.objectives
            return False

        return False

    def initFitness(self):
        return [np.inf if direction == -1 else 0 for direction in self.problem.directions]


class _RandomIntGenerator(Generator):

    def __init__(self):
        super(_RandomIntGenerator, self).__init__()

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


# Break apart the grid.matrix array
class EncoderDecoder(object):
    def __init__(self, model):

        self.encoded = False
        self.rawModel = model
        self.rebuiltModel = None

        self.removedAttributesList = ['matrixA', 'matrixB', 'matrixX']

        # These members are the dictionary results of the json serialization format required in json.dump
        self.encodedAttributes = {}

        self.typeList = []
        self.unacceptedTypeList = [complex, np.complex128]

        # This dict contains the attributes that contain unaccepted objects as values
        self.unacceptedJsonAttributes = {'matrix': Node, 'errorDict': (TransformedDict, Error),
                                         'hmUnknownsList': Region, 'terminalSlots': dict}
        self.unacceptedJsonClasses = {Material}
        self.unacceptedJsonObjects = []
        self.__setFlattenJsonObjects()

    def __setFlattenJsonObjects(self):
        for value in self.unacceptedJsonAttributes.values():
            try:
                for nestedVal in value:
                    self.unacceptedJsonObjects.append(nestedVal)
            except TypeError:
                self.unacceptedJsonObjects.append(value)

    def __convertComplexInList(self, inList):
        # 'plex_Signature' is used to identify if a list is a destructed complex number or not
        return list(map(lambda x: ['plex_Signature', x.real, x.imag], inList))

    def __objectToDict(self, attribute, originalValue):
        if attribute == 'errorDict':
            errorDict = originalValue.__dict__['store']
            returnVal = {key: error.__dict__ for key, error in errorDict.items()}
        elif attribute == 'matrix':
            listedMatrix = originalValue.tolist()
            # Case 2D list
            for idxRow, row in enumerate(listedMatrix):
                for idxCol, col in enumerate(row):
                    listedMatrix[idxRow][idxCol] = {key: value if type(value) not in self.unacceptedTypeList else ['plex_Signature', value.real, value.imag] for key, value in row[idxCol].__dict__.items()}
            returnVal = listedMatrix
        elif attribute == 'hmUnknownsList':
            returnVal = {}
            tempDict = {}
            for regionKey, regionVal in originalValue.items():
                for dictKey, dictVal in regionVal.__dict__.items():
                    if dictKey not in ['an', 'bn']:
                        tempDict[dictKey] = dictVal
                    else:
                        tempDict[dictKey] = self.__convertComplexInList(dictVal.tolist()) if type(dictVal) == np.ndarray else dictVal
                returnVal[regionKey] = tempDict
        elif type(originalValue) == Material:
            returnVal = {key: _property for key, _property in originalValue.__dict__.items()}
        else:
            print('This attribute was not encoded: ', attribute)
            returnVal = None

        return returnVal

    def __filterValType(self, inAttr, inVal):

        # Json dump cannot handle user defined class objects
        if inAttr in self.unacceptedJsonAttributes or type(inVal) in self.unacceptedJsonClasses:
            outVal = self.__objectToDict(inAttr, inVal)

        # Json dump cannot handle numpy arrays
        elif type(inVal) == np.ndarray:
            if inVal.size == 0:
                outVal = []
            else:
                outVal = inVal.tolist()
                # Case 1D list containing unacceptedTypes
                if type(outVal[0]) in self.unacceptedTypeList:
                    outVal = self.__convertComplexInList(outVal)

        # Json dump and load cannot handle complex types
        elif type(inVal) in self.unacceptedTypeList:
            # 'plex_Signature' is used to identify if a list is a destructed complex number or not
            outVal = ['plex_Signature', inVal.real, inVal.imag]

        else:
            outVal = inVal

        return outVal

    def __buildTypeList(self, val):
        # Generate a list of data types before destructing the matrix
        if type(val) not in self.typeList:
            self.typeList.append(type(val))

    def __getType(self, val):
        return str(type(val)).split("'")[1]

    # Encode all attributes that have valid json data types
    def __destruct(self):
        self.encodedAttributes = {}
        for attr, val in self.rawModel.__dict__.items():
            # Matrices included in the set of linear equations were excluded due to computation considerations
            if attr not in self.removedAttributesList:
                self.__buildTypeList(val)
                self.encodedAttributes[attr] = self.__filterValType(attr, val)

    # Rebuild objects that were deconstructed to store in JSON object
    def __construct(self):

        errors = self.encodedAttributes['errorDict']
        matrix = self.encodedAttributes['matrix']
        hmUnknowns = self.encodedAttributes['hmUnknownsList']

        # ErrorDict reconstruction
        self.encodedAttributes['errorDict'] = TransformedDict.buildFromJson(errors)

        # Matrix reconstruction
        Cnt = 0
        lenKeys = len(matrix)*len(matrix[0])
        constructedMatrix = np.array([type('', (Node,), {}) for _ in range(lenKeys)])
        for row in matrix:
            for nodeInfo in row:
                constructedMatrix[Cnt] = Node.buildFromJson(nodeInfo)
                Cnt += 1
        rawArrShape = self.rawModel.matrix.shape
        constructedMatrix = constructedMatrix.reshape(rawArrShape[0], rawArrShape[1])
        self.encodedAttributes['matrix'] = constructedMatrix

        # HmUnknownsList Reconstruction
        self.encodedAttributes['hmUnknownsList'] = {i: Region.rebuildFromJson(jsonObject=hmUnknowns[i]) for i in self.encodedAttributes['hmRegions']}

        self.rebuiltModel = Model.buildFromJson(jsonObject=self.encodedAttributes)

    def jsonStoreSolution(self):

        self.__destruct()
        # Data written to file
        if not os.path.isdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        with open(DATA_PATH, 'w') as StoredSolutionData:
            json.dump(self.encodedAttributes, StoredSolutionData)
        # Data read from file
        with open(DATA_PATH) as StoredSolutionData:
            # json.load returns dicts where keys are converted to strings
            self.encodedAttributes = json.load(StoredSolutionData)

        self.__construct()
        if self.rawModel.equals(self.rebuiltModel, self.removedAttributesList):
            self.encoded = True
        else:
            self.rebuiltModel.writeErrorToDict(key='name',
                                               error=Error.buildFromScratch(name='JsonRebuild',
                                                                            description="ERROR - Model Did Not Rebuild Correctly",
                                                                            cause=True))


def buildMotor(motorCfg, hamCfg, canvasCfg, run=False, baseline=False, optimize=False):
    '''
    This is a function that allows for a specific motor configuration without optimization to be simulated
    '''

    if not run:
        return

    # Object for the model design, grid, and matrices
    if baseline:
        model = Model.buildBaseline(motorCfg=motorCfg, hamCfg=hamCfg, canvasCfg=canvasCfg)
    else:
        model = Model.buildFromScratch(motorCfg=motorCfg, hamCfg=hamCfg, canvasCfg=canvasCfg)

    model.buildGrid()
    model.checkSpatialMapping()
    errorInX = model.finalizeCompute()
    # TODO This invertY inverts the pyplot
    model.updateGrid(errorInX, canvasCfg=canvasCfg, invertY=True)
    # This is done for computation considerations within the optimization loop
    if optimize and model.errorDict.isEmpty():
        return model

    # After this point, the json implementations should be used to not branch code direction
    encodeModel = EncoderDecoder(model)
    encodeModel.jsonStoreSolution()

    if encodeModel.rebuiltModel.errorDict.isEmpty():
        # iDims (height x width): BenQ = 1440 x 2560, ViewSonic = 1080 x 1920
        # model is only passed in to showModel to show the matrices A and B since they are not stored in the json object
        # TODO This invertY inverts the Tkinter Canvas plot
        LIM.Show.showModel(encodeModel, model, canvasCfg, numColours=20, dims=[1080, 1920], invertY=False)
        print('   - there are no errors')
        return model
    else:
        print('\nBelow are a list of warnings and errors:')
        print('*) Resolve errors to show model')
        encodeModel.rebuiltModel.errorDict.printErrorsByAttr('description')


def solveOptimization(algorithm, problem, solverList, constraint, termination, solver, run=False):
    if not run:
        return solverList
    prob = problem(**constraint)
    optimization = algorithm(prob, **solver)
    optimization.run(_TerminationCondition(prob, **termination))
    if hasattr(algorithm, '__name__'):
        return addAlgoName(solverList, algorithm.__name__)
    else:
        return addAlgoName(solverList, type(algorithm).__name__)


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
    solverList = solveOptimization(GeneticAlgorithm, SchwefelProblem, solverList, constraint_params, termination_params,
                                   ga_params, run=True)

    # TODO NSGA doesnt have offspring parameter because it clips to population size instead
    #  -> if pop = 200 then offspring + pop = 400 which is fitness sorted and clipped to 200 again
    nsga_params = {'population_size': parent_size, 'generator': RandomGenerator(),
                   'selector': TournamentSelector(tournament_size), 'variator': GAOperator(SBX(0.3), PM(0.1)),
                   'archive': FitnessArchive(nondominated_sort)}
    solverList = solveOptimization(NSGAII, SchwefelProblem, solverList, constraint_params, termination_params,
                                   nsga_params, run=True)

    pso_params = {'swarm_size': parent_size, 'leader_size': leader_size, 'generator': RandomGenerator(),
                  'mutate': PM(0.1), 'leader_comparator': AttributeDominance(objective_key),
                  'larger_preferred': True, 'fitness': crowding_distance, 'fitness_getter': objective_key}
    solverList = solveOptimization(_ParticlSwarm, SchwefelProblem, solverList, constraint_params, termination_params,
                                   pso_params, run=True)

    omopso_params = {'epsilons': [0.05], 'swarm_size': parent_size, 'leader_size': leader_size,
                     'mutation_probability': 0.1, 'mutation_perturbation': 0.5, 'max_iterations': 100,
                     'generator': RandomGenerator(), 'selector': TournamentSelector(tournament_size),
                     'variator': SBX(0.3)}
    solverList = solveOptimization(OMOPSO, SchwefelProblem, solverList, constraint_params, termination_params,
                                   omopso_params, run=True)

    solutions = AbstractProblem.getSolutionFromLog(solverList)

    summarizedResult = SchwefelProblem.plottingPerformance(solverList, solutions, plot=True)
    for name, generations in summarizedResult.items():
        print(f"________{name}________\n"
              f"objective: {generations['best_objectives'][-1]}\n"
              f"variables: {generations['best_variables'][-1]}\n"
              f"nfe: {generations['nfe']}\n"
              f"time: {generations['time']}\n")
    SchwefelProblem.plottingConvergence(x1, x2, lower, upper, solutions, run=True)
    SchwefelProblem.plottingSchwefel(x1, x2, lower, upper, run=True)


if __name__ == '__main__':
    main()
