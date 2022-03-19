import numpy as np

from LIM.Show import *
import json
import os

PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Output')
DATA_PATH = os.path.join(OUTPUT_PATH, 'StoredSolutionData.json')

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
'''


# Problem is the object that Test will inherit from
# class Test(Problem):
#
#     def __init__(self, iN):
#         self.n = iN
#         # The numbers indicate: #inputs, #objectives, #constraints
#         super(Test, self).__init__(2, 2, 1)
#         # Constrain the range and type for each input
#         self.types[:] = [Integer(12, 50), Integer(12, 50)]
#         # Constrain every input. This works in unison with with constraints in the evaluate method
#         # Ex) Slots >= Poles becomes Slots - Poles >= 0 so init constraint becomes ">=0" and evaluate constraint becomes Slots - Poles
#         self.constraints[:] = ">=0"
#         # Choose which objective to maximize and minimize
#         self.directions[:] = [self.MINIMIZE, self.MAXIMIZE]
#
#     def evaluate(self, solution):
#         slots = solution.variables[0]
#         poles = solution.variables[1]
#
#         q = slots / poles / 3
#         if slots > poles and slots > 6 and poles % 2 == 0 and q % 1 == 0 and q != 0:
#             motor = Motor(slots, poles, length)
#             grid = HAM_Grid(iDesign=motor, iN=self.n, iCanvasSpacing=10000/motor.H, iPixelDivision=15)
#             grid, matrixX = HAM_Compute(iDesign=motor, iGrid=grid, iN=self.n)
#             grid = HAM_UpdateGrid(iDesign=motor, iGrid=grid, iMatrixX=matrixX, iN=self.n)
#             mass_fitness = motor.MassTot
#             thrust_fitness = grid.Fx
#         else:
#             mass_fitness = math.inf
#             thrust_fitness = 0
#
#         print('solution: ', slots, poles, mass_fitness, thrust_fitness)
#
#         solution.objectives[:] = [mass_fitness, thrust_fitness]
#         solution.constraints[:] = [slots - poles]


# Break apart the grid.matrix array

class EncoderDecoder(object):
    def __init__(self, model):

        self.encoded = False
        # After instantiating seld.model with model. this should be used rather than the old model
        # TODO There could be an issue with self.complexList and self.model.complexList value difference because
        self.rawModel = model
        self.rebuiltModel = None

        self.unacceptedJsonAttributes = ['matrix', 'errorDict', 'hmUnknownsList']
        # These members are the dictionary results of the json serialization format required in json.dump
        self.encodedAttributes = {}
        self.encodedMatrix = {}
        self.encodedErrorDict = {}
        self.encodedHmUnknownsList = {}

        self.typeList = []
        self.unacceptedTypeList = [complex, np.complex128]

    def filterValType(self, val):
        # JSON dump cannot handle numpy arrays
        if type(val) == np.ndarray:
            val = val.tolist()
            # these lists can still contain unacceptable types so I need to filter that. Use the map or filter methods
            # I should also replace all instances of list(ndarray) to ndarray.tolist()

        # Json dump and load cannot handle complex objects and must be an accepted dtype such as list
        elif type(val) in self.unacceptedTypeList:
            # 'plex_Signature' is used to identify if a list is a destructed complex number or not
            val = ['plex_Signature', val.real, val.imag]

        else:
            pass

        return val

    def __buildTypeList(self, val):
        # Generate a list of data types before destructing the matrix
        if self.getType(val) not in self.typeList:
            self.typeList += [self.getType(val)]

    def getType(self, val):
        return str(type(val)).split("'")[1]

    # Encode all attributes that have valid json data types
    def encodeAttributes(self):
        self.encodedAttributes = {}
        # TODO I need to make a filter here turn ndarray into list - IS A LOT OF WORK
        # InfoRes includes all the grid info that is serializable for a JSON object
        for attr, val in self.rawModel.__dict__.items():
            if attr not in self.unacceptedJsonAttributes:
                self.__buildTypeList(val)
                self.encodedAttributes[attr] = self.filterValType(val)

    # Encode a 2D matrix of Node objects
    def encodeMatrix(self):
        Cnt = 0
        for row in self.rawModel.matrix:
            for node in row:
                nodeDictList = []
                for attr, val in node.__dict__.items():

                    self.__buildTypeList(val)
                    val = self.filterValType(val)
                    nodeDictList.append((attr, val))

                self.encodedMatrix[Cnt] = dict(nodeDictList)

                Cnt += 1

    # Encode a 1D TransformedDict of Error objects
    def encodeErrorDict(self):
        for error_name in self.rawModel.errorDict:
            listDict = {}
            for error_member in self.rawModel.errorDict[error_name].__dict__:

                val = self.rawModel.errorDict[error_name].__dict__[error_member]
                self.__buildTypeList(val)
                val = self.filterValType(val)
                listDict[error_member] = val

            self.encodedErrorDict[error_name] = listDict

    # Encode a 1D list of Region objects
    def encodeHmUnknownsList(self):
        for regionNum in self.rawModel.hmUnknownsList:  # iterate through Regions
            listDict = {}
            for key in self.rawModel.hmUnknownsList[regionNum].__dict__:  # iterate through .an and .bn per Region
                assigned = self.rawModel.hmUnknownsList[regionNum].__dict__[key]
                valDict = {}
                if type(assigned) in [list, np.ndarray]:
                    for countVal, val in enumerate(assigned):  # iterate through arrays of .an or .bn
                        self.__buildTypeList(val)
                        val = self.filterValType(val)
                        valDict[countVal] = val

                    listDict[key] = valDict
                else:
                    listDict[key] = assigned

            self.encodedHmUnknownsList[regionNum] = listDict

    def destruct(self):

        self.encodeAttributes()
        self.encodeMatrix()
        self.encodeErrorDict()
        self.encodeHmUnknownsList()

        print(f'typeList: {self.typeList}, unacceptedTypeList: {self.unacceptedTypeList}')

        attributesDict = {'attributes': self.encodedAttributes, 'matrix': self.encodedMatrix,
                          'errorDict': self.encodedErrorDict, 'hmUnknownsList': self.encodedHmUnknownsList}

        return attributesDict

    # Rebuild objects that were deconstructed to store in JSON object
    def construct(self, iDict):

        attributes = iDict['attributes']
        matrix = iDict['matrix']
        errors = iDict['errorDict']
        hmUnknowns = iDict['hmUnknownsList']

        # Matrix Reconstruction
        lenKeys = len(dict.keys(matrix))
        constructedMatrix = np.array([type('', (object,), {}) for _ in range(lenKeys)])
        for key in dict.keys(matrix):
            nodeInfo = matrix[key]
            emptyNode = Node()
            rebuiltNode = emptyNode.rebuildNode(nodeInfo)
            constructedMatrix[int(key)] = rebuiltNode

        rawArrShape = self.rawModel.matrix.shape
        constructedMatrix = constructedMatrix.reshape(rawArrShape[0], rawArrShape[1])

        # HmUnknownsList Reconstruction
        constructedHmUnknowns = {i: Region.rebuildFromJson(jsonObject=hmUnknowns[i]) for i in attributes['hmRegions']}

        self.rebuiltModel = Model.buildFromJson(jsonObject={'attributes': attributes, 'matrix': constructedMatrix,
                                                            'hmUnknowns': constructedHmUnknowns})

        # ErrorDict Reconstruction
        for error_name in errors:
            self.rebuiltModel.writeErrorToDict(key='name',
                                               error=Error.buildFromJson(jsonObject=errors[error_name]))

    def jsonStoreSolution(self):

        destructed = self.destruct()
        # Data written to file
        if not os.path.isdir(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
        with open(DATA_PATH, 'w') as StoredSolutionData:
            json.dump(destructed, StoredSolutionData)
        # Data read from file
        with open(DATA_PATH) as StoredSolutionData:
            dictionary = json.load(StoredSolutionData)

        self.construct(iDict=dictionary)
        # TODO I should override the equals method of model to check that the entire object is the same not just matrix
        if self.rebuiltModel == self.rawModel:
            self.encoded = True
        checkIdenticalLists = np.array([[self.rawModel.matrix[y, x] == self.rebuiltModel.matrix[y, x] for x in range(self.rebuiltModel.ppL)] for y in range(self.rebuiltModel.ppH)])

        if np.all(np.all(checkIdenticalLists, axis=1)):
            self.encoded = True
        else:
            self.rebuiltModel.writeErrorToDict(key='name',
                                               error=Error.buildFromScratch(name='gridMatrixJSON',
                                                                            description="ERROR - The JSON object matrix does not match the original matrix",
                                                                            cause=True))


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
    # Efficient to simulate at pixDiv >= 10, but fastest at pixDiv = 2
    pixelDivisions = 5

    lowDiscrete = 50
    # n list does not include n = 0 harmonic since the average of the complex fourier series is 0,
    # since there are no magnets or anything constantly creating a magnetic field when input is off
    n = range(-lowDiscrete, lowDiscrete + 1)
    n = np.delete(n, len(n)//2, 0)
    slots = 16
    poles = 6
    wt, ws = 6 / 1000, 10 / 1000
    slotpitch = wt + ws
    endTeeth = 2 * (5/3 * wt)
    length = ((slots - 1) * slotpitch + ws) + endTeeth

    # with timing():
    #     algorithm = NSGAII(Test(n))
    #     algorithm.run(5000)
    #
    # # # TODO Platypus has parallelization built in instead of using Jit
    # # # TODO Look into changing population
    # feasible_solutions = [s for s in algorithm.result if s.feasible]
    # # # TODO Use the debugging tool to see the attributes of algorithm.population
    # plotResults = []
    # cnt = 1
    # for results in feasible_solutions:
    #     slotpoles = [algorithm.problem.types[0].decode(results.variables[0]), algorithm.problem.types[0].decode(results.variables[1])]
    #     objectives = [results.objectives[0], results.objectives[1]]
    #     # TODO Think of a good way to plot results
    #     print(slotpoles, objectives)
    #     plotResults.append((slotpoles, objectives))
    #     cnt += 1

    # fig, axs = plt.subplots(2)
    # fig.suptitle('Mass and Thrust')
    # axs[0].scatter(range(1, cnt), [x[1, 0] for x in plotResults])
    # axs[0].set_title('Mass')
    # axs[1].scatter(range(1, cnt), [x[1, 1] for x in plotResults])
    # axs[1].set_title('Thrust')
    # axs[2].scatter(range(1, cnt), [x[0, 0] for x in plotResults])
    # axs[2].set_title('mass obj')
    # axs[3].scatter(range(1, cnt), [x[0, 1] for x in plotResults])
    # axs[3].set_title('thrust obj')
    # plt.scatter(range(1, cnt), [x[1, 1] for x in pltResults])
    # # plt.scatter(range(1, cnt), [x[1, 1] for x in plotResults])
    # plt.xlabel('Generations')
    # plt.ylabel('Thrust Objective Value')
    # plt.show()
    # I need to fix the relative infinity in matrix A

    # Used for testing
    # for keys in tempMotor.__dict__.items():
    #     print(keys)

    # This value defines how small the mesh is at [border, border+1]. Ex) [4, 2] means that the mesh at the border is 1/4 the mesh far away from the border
    meshDensity = np.array([4, 2])
    # Change the mesh density at boundaries. A 1 indicates denser mesh at a size of len(meshDensity)
    # [LeftAirBuffer], [LeftEndTooth], [Slots], [FullTeeth], [LastSlot], [RightEndTooth], [RightAirBuffer]

    xMeshIndexes = [[0, 0]] + [[0, 0]] + [[0, 0], [0, 0]] * (slots - 1) + [[0, 0]] + [[0, 0]] + [[0, 0]]
    # [LowerVac], [Yoke], [LowerSlots], [UpperSlots], [Airgap], [BladeRotor], [BackIron], [UpperVac]
    yMeshIndexes = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

    pixelSpacing = slotpitch/pixelDivisions
    canvasSpacing = 80

    # Object for the model design, grid, and matrices
    model = Model.buildFromScratch(slots=slots, poles=poles, length=length, n=n,
                                   pixelSpacing=pixelSpacing, canvasSpacing=canvasSpacing,
                                   meshDensity=meshDensity, meshIndexes=[xMeshIndexes, yMeshIndexes],
                                   hmRegions=
                                   {1: 'vac', 2: 'bi', 3: 'dr', 4: 'g', 6: 'vac'},
                                   mecRegions=
                                   {5: 'core_1'})

    model.buildGrid(pixelSpacing, [xMeshIndexes, yMeshIndexes])
    model.finalizeGrid(pixelDivisions)

    with timing():
        errorInX = model.finalizeCompute(iTol=1e-15)

    model.updateGrid(errorInX, showAirgapPlot=False)

    # After this point, the json implementations should be used to not branch code direction
    encodeModel = EncoderDecoder(model)
    encodeModel.jsonStoreSolution()

    if encodeModel.rebuiltModel.errorDict.isEmpty():
        # iDims (height x width): BenQ = 1440 x 2560, ViewSonic = 1080 x 1920
        # showModel(gridInfo, gridMatrix, model, fieldType='Yk',
        #           showGrid=True, showFields=True, showFilter=False, showMatrix=False, showZeros=True,
        #           numColours=20, dims=[1080, 1920])
        pass
    else:
        print('Resolve errors to show model')

    print('\nBelow are a list of warnings and errors:')
    if encodeModel.rebuiltModel.errorDict:
        encodeModel.rebuiltModel.errorDict.printErrorsByAttr('description')
    else:
        print('   - there are no errors')


if __name__ == '__main__':
    # profile_main()  # To profile the main execution
    main()
