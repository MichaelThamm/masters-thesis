import math

import matplotlib.pyplot as plt

from Optimizations.OptimizationCfg import *
from LIM.Show import *
from platypus import *
import json
import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Output')
DATA_PATH = os.path.join(OUTPUT_PATH, 'StoredSolutionData.json')

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
'''


class MotorOptProblem(Problem):

    def __init__(self, slots, poles):
        # The numbers indicate: #inputs, #objectives, #constraints
        super(MotorOptProblem, self).__init__(2, 2, 1)
        # Constrain the range and type for each input
        self.types[:] = [Integer(slots[0], slots[1]), Integer(poles[0], poles[1])]
        # Constrain every input. This works in unison with with constraints in the evaluate method
        # Ex) Slots >= Poles becomes Slots - Poles >= 0 so init constraint becomes ">=0" and evaluate constraint becomes Slots - Poles
        self.constraints[:] = ">=0"
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE, self.MAXIMIZE]

    def evaluate(self, solution):
        # TODO To get this to work you need to uncomment the robust ratios in SlotPoleCalculation
        slots = solution.variables[0]
        poles = solution.variables[1]

        q = slots / poles / self.m
        print(slots, poles)
        # if slots > poles and slots > 6 and poles % 2 == 0 and q % 1 == 0 and q != 0:
        if slots > poles and slots > 6 and poles % 2 == 0:

            pixelDivisions = 2
            lowDiscrete = 50
            n = range(-lowDiscrete, lowDiscrete + 1)
            n = np.delete(n, len(n) // 2, 0)
            # TODO A couple of interesting things to note on failing models is that the dimensions of matrix A is 1001x1001 which should not be possible
            #  Also there is a spatial error with flag - iGrid spacing
            #  I can take the inputs that give me a hard time and use them in the baselineMotor to test them ex) (30, 26), (27, 22), (49, 40)
            length = 0.27
            slotpitch = length / slots

            meshDensity = np.array([4, 2])
            xMeshIndexes = [[0, 0]] + [[0, 0]] + [[0, 0], [0, 0]] * (slots - 1) + [[0, 0]] + [[0, 0]] + [[0, 0]]
            yMeshIndexes = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
            pixelSpacing = slotpitch / pixelDivisions
            canvasSpacing = 80

            regionCfg1 = {'hmRegions': {1: 'vac_lower', 2: 'bi', 3: 'dr', 4: 'g', 6: 'vac_upper'},
                          'mecRegions': {5: 'mec'},
                          'invertY': False}

            choiceRegionCfg = regionCfg1

            # Object for the model design, grid, and matrices
            model = Model.buildFromScratch(slots=slots, poles=poles, length=length, n=n,
                                           pixelSpacing=pixelSpacing, canvasSpacing=canvasSpacing,
                                           meshDensity=meshDensity, meshIndexes=[xMeshIndexes, yMeshIndexes],
                                           hmRegions=
                                           choiceRegionCfg['hmRegions'],
                                           mecRegions=
                                           choiceRegionCfg['mecRegions'],
                                           errorTolerance=1e-15,
                                           # If invertY = False -> [LowerSlot, UpperSlot, Yoke]
                                           # TODO This invertY flips the core MEC region
                                           invertY=choiceRegionCfg['invertY'])

            model.buildGrid(pixelSpacing, [xMeshIndexes, yMeshIndexes])
            model.finalizeGrid(pixelDivisions)
            errorInX = model.finalizeCompute()
            model.updateGrid(errorInX, showAirgapPlot=False, invertY=True, showUnknowns=False)

            mass_fitness = model.MassTot
            thrust_fitness = model.Fx

        else:
            mass_fitness = math.inf
            thrust_fitness = 0

        # print('solution: ', slots, poles, mass_fitness, thrust_fitness)

        solution.objectives[:] = [mass_fitness, thrust_fitness]
        solution.constraints[:] = [slots - poles]


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
        self.unacceptedJsonAttributes = {'matrix': Node, 'errorDict': (TransformedDict, Error), 'hmUnknownsList': Region}
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
        else:
            print('The object does not match a conditional')
            return

        return returnVal

    def __filterValType(self, inAttr, inVal):

        # Json dump cannot handle user defined class objects
        if inAttr in self.unacceptedJsonAttributes:
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


def platypus(run=False):
    if not run:
        return

    logging.FileHandler(LOGGER_FILE, mode='w')

    tolerance = 10 ** (-7)
    max_evals = 30000 - 1
    max_stalls = 25
    stall_tolerance = tolerance
    timeout = 3000000  # seconds
    parent_size = 200
    tournament_size = 2
    constraint_params = {'slots': [10, 24], 'poles': [4, 9]}
    termination_params = {'max_evals': max_evals, 'tolerance': tolerance,
                          'max_stalls': max_stalls, 'stall_tolerance': stall_tolerance,
                          'timeout': timeout}
    solverList = []

    # TODO There is an issue with the initialization of variables. It seems like the support for Integer is not really there
    #   I can try to fix this but swap to Real instead and then just round in the evaluate method instead using floor div
    nsga_params = {'population_size': parent_size, 'generator': RandomGenerator(),
                   'selector': TournamentSelector(tournament_size), 'variator': GAOperator(SBX(0.3), PM(0.1)),
                   'archive': FitnessArchive(nondominated_sort)}
    solverList = solveOptimization(NSGAII, MotorOptProblem, solverList, constraint_params, termination_params, nsga_params, run=True)


def baselineMotor(run=False):
    if not run:
        return

    # Efficient to simulate at pixDiv >= 10, but fastest at pixDiv = 2
    pixelDivisions = 2

    lowDiscrete = 50
    # n list does not include n = 0 harmonic since the average of the complex fourier series is 0,
    # since there are no magnets or anything constantly creating a magnetic field when input is off
    n = range(-lowDiscrete, lowDiscrete + 1)
    n = np.delete(n, len(n)//2, 0)
    slots = 16
    poles = 6
    length = 0.27
    slotpitch = length / slots

    # This value defines how small the mesh is at [border, border+1].
    # Ex) [4, 2] means that the mesh at the border is 1/4 the mesh far away from the border
    meshDensity = np.array([4, 2])
    # Change the mesh density at boundaries. A 1 indicates denser mesh at a size of len(meshDensity)
    # [LeftAirBuffer], [LeftEndTooth], [Slots], [FullTeeth], [LastSlot], [RightEndTooth], [RightAirBuffer]

    xMeshIndexes = [[0, 0]] + [[0, 0]] + [[0, 0], [0, 0]] * (slots - 1) + [[0, 0]] + [[0, 0]] + [[0, 0]]
    # TODO yMeshIndexes needs to incorporate invertY and the removal of Dirichlet indexes
    # [LowerVac], [Yoke], [LowerSlots], [UpperSlots], [Airgap], [BladeRotor], [BackIron], [UpperVac]
    yMeshIndexes = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    pixelSpacing = slotpitch/pixelDivisions
    canvasSpacing = 80

    regionCfg1 = {'hmRegions': {1: 'vac_lower', 2: 'bi', 3: 'dr', 4: 'g', 6: 'vac_upper'},
                  'mecRegions': {5: 'mec'},
                  'invertY': False}
    # TODO Note that Cfg2 is wrong since the coil pattern in the x direction is only intended for Cfg1
    #  however, I could write some code to loop through all rows of matrix and invert them and test the results
    regionCfg2 = {'hmRegions': {1: 'vac_lower', 3: 'g', 4: 'dr', 5: 'bi', 6: 'vac_upper'},
                  'mecRegions': {2: 'mec'},
                  'invertY': True}

    # TODO This errors because parts of the code are not linked to hmmecRegions since self.ppH doesnt change
    regionCfg3 = {'hmRegions': {1: 'vac_lower', 2: 'bi', 3: 'dr', 5: 'vac_upper'},
                  'mecRegions': {4: 'mec'},
                  'invertY': False}

    regionCfg4 = {'hmRegions': {},
                  'mecRegions': {1: 'mec'},
                  'invertY': False}

    choiceRegionCfg = regionCfg1

    # Object for the model design, grid, and matrices
    model = Model.buildBaseline(slots=slots, poles=poles, length=length, n=n,
                                   pixelSpacing=pixelSpacing, canvasSpacing=canvasSpacing,
                                   meshDensity=meshDensity, meshIndexes=[xMeshIndexes, yMeshIndexes],
                                   hmRegions=
                                   choiceRegionCfg['hmRegions'],
                                   mecRegions=
                                   choiceRegionCfg['mecRegions'],
                                   errorTolerance=1e-14,
                                   # If invertY = False -> [LowerSlot, UpperSlot, Yoke]
                                   # TODO This invertY flips the core MEC region
                                   invertY=choiceRegionCfg['invertY'])

    model.buildGrid(pixelSpacing, [xMeshIndexes, yMeshIndexes])
    model.finalizeGrid(pixelDivisions)

    with timing():
        errorInX = model.finalizeCompute()

    # TODO This invertY inverts the pyplot
    model.updateGrid(errorInX, showAirgapPlot=True, invertY=True, showUnknowns=False)

    # After this point, the json implementations should be used to not branch code direction
    encodeModel = EncoderDecoder(model)
    encodeModel.jsonStoreSolution()

    # TODO The or True is here for convenience but should be removed
    if encodeModel.rebuiltModel.errorDict.isEmpty() or True:
        # iDims (height x width): BenQ = 1440 x 2560, ViewSonic = 1080 x 1920
        # model is only passed in to showModel to show the matrices A and B since they are not stored in the json object
        showModel(encodeModel, model, fieldType='B',
                  showGrid=True, showFields=True, showFilter=False, showMatrix=False, showZeros=True,
                  # TODO This invertY inverts the Tkinter Canvas plot
                  numColours=20, dims=[1080, 1920], invertY=False)
        pass
    else:
        print('Resolve errors to show model')

    print('\nBelow are a list of warnings and errors:')
    if encodeModel.rebuiltModel.errorDict:
        encodeModel.rebuiltModel.errorDict.printErrorsByAttr('description')
    else:
        print('   - there are no errors')


def buildMotor(run=False):
    '''
    This is a function that allows for a specific motor configuration without optimization to be simulated
    '''
    if not run:
        return

    # Efficient to simulate at pixDiv >= 10, but fastest at pixDiv = 2
    pixelDivisions = 2

    lowDiscrete = 50
    # n list does not include n = 0 harmonic since the average of the complex fourier series is 0,
    # since there are no magnets or anything constantly creating a magnetic field when input is off
    n = range(-lowDiscrete, lowDiscrete + 1)
    n = np.delete(n, len(n)//2, 0)
    slots = 21
    poles = 6
    length = 0.27
    slotpitch = length / slots

    meshDensity = np.array([4, 2])
    xMeshIndexes = [[0, 0]] + [[0, 0]] + [[0, 0], [0, 0]] * (slots - 1) + [[0, 0]] + [[0, 0]] + [[0, 0]]
    yMeshIndexes = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    pixelSpacing = slotpitch / pixelDivisions
    canvasSpacing = 80

    regionCfg1 = {'hmRegions': {1: 'vac_lower', 2: 'bi', 3: 'dr', 4: 'g', 6: 'vac_upper'},
                  'mecRegions': {5: 'mec'},
                  'invertY': False}

    choiceRegionCfg = regionCfg1

    # Object for the model design, grid, and matrices
    model = Model.buildFromScratch(slots=slots, poles=poles, length=length, n=n,
                                   pixelSpacing=pixelSpacing, canvasSpacing=canvasSpacing,
                                   meshDensity=meshDensity, meshIndexes=[xMeshIndexes, yMeshIndexes],
                                   hmRegions=choiceRegionCfg['hmRegions'],
                                   mecRegions=choiceRegionCfg['mecRegions'],
                                   errorTolerance=1e-15,
                                   # If invertY = False -> [LowerSlot, UpperSlot, Yoke]
                                   invertY=choiceRegionCfg['invertY'])

    model.buildGrid(pixelSpacing, [xMeshIndexes, yMeshIndexes])
    model.finalizeGrid(pixelDivisions)
    errorInX = model.finalizeCompute()
    model.updateGrid(errorInX, showAirgapPlot=False, invertY=True, showUnknowns=False)

    # After this point, the json implementations should be used to not branch code direction
    encodeModel = EncoderDecoder(model)
    encodeModel.jsonStoreSolution()

    # TODO The or True is here for convenience but should be removed
    if encodeModel.rebuiltModel.errorDict.isEmpty() or True:
        # iDims (height x width): BenQ = 1440 x 2560, ViewSonic = 1080 x 1920
        # model is only passed in to showModel to show the matrices A and B since they are not stored in the json object
        showModel(encodeModel, model, fieldType='B',
                  showGrid=True, showFields=True, showFilter=False, showMatrix=False, showZeros=True,
                  # TODO This invertY inverts the Tkinter Canvas plot
                  numColours=20, dims=[1080, 1920], invertY=False)
        pass
    else:
        print('Resolve errors to show model')

    print('\nBelow are a list of warnings and errors:')
    if encodeModel.rebuiltModel.errorDict:
        encodeModel.rebuiltModel.errorDict.printErrorsByAttr('description')
    else:
        print('   - there are no errors')


def profile_main():

    import cProfile, pstats, io

    prof = cProfile.Profile()
    prof = prof.runctx("main()", globals(), locals())

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

    platypus(run=False)
    baselineMotor(run=False)
    # TODO You can see the winding pattern is different when using 21 slots, 6 poles. Need consistency
    buildMotor(run=True)


if __name__ == '__main__':
    # profile_main()  # To profile the main execution
    # main()
    resistivity = 1/(17*10**6)
    # resistivity = 2.857*10**-8
    print(resistivity)
    frequency = np.arange(1, 170, 1)
    mu = 1
    skin_depth = 1000*np.sqrt(2*resistivity/(frequency*mu))
    skin_depth_1 = 1000*np.sqrt(2*resistivity/(1*mu))
    skin_depth_10 = 1000*np.sqrt(2*resistivity/(10*mu))
    print(skin_depth_1, skin_depth_10)
    print(skin_depth)
    plt.plot(frequency, skin_depth)
    # TODO Throw this in a function and make it look like swiss. Then figure out why I cant reproduce swiss
    plt.axhline((max(skin_depth) - min(skin_depth))/2)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Skin Depth (mm)')
    plt.title('Skin Depth vs. Frequency')
    plt.show()
