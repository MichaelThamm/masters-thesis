from builtins import list
from numpy.core._multiarray_umath import ndarray
from LIM.SlotPoleCalculation import*
from collections import deque

PP_SLOTHEIGHT = 'ppSlotHeight'


class Grid(LimMotor):

    lower_slotsA: ndarray
    upper_slotsA: ndarray
    lower_slotsB: ndarray
    upper_slotsB: ndarray
    lower_slotsC: ndarray
    upper_slotsC: ndarray

    inLower_slotsA: ndarray
    outLower_slotsA: ndarray
    inUpper_slotsA: ndarray
    outUpper_slotsA: ndarray
    inLower_slotsB: ndarray
    outLower_slotsB: ndarray
    inUpper_slotsB: ndarray
    outUpper_slotsB: ndarray
    inLower_slotsC: ndarray
    outLower_slotsC: ndarray
    inUpper_slotsC: ndarray
    outUpper_slotsC: ndarray

    def __init__(self, kwargs, buildBaseline=False):

        super().__init__(kwargs['motorCfg'], buildBaseline)

        # n list does not include n = 0 harmonic since the average of the complex fourier series is 0,
        #  since there are no magnets or anything constantly creating a magnetic field when input is off
        harmBounds = kwargs['hamCfg']['N'] // 2 if kwargs['hamCfg']['N'] % 2 == 0 else (kwargs['hamCfg']['N'] + 1) // 2
        n = range(-harmBounds, harmBounds + 1)
        self.n = np.delete(n, len(n) // 2, 0)

        # Turn inputs into attributes
        self.invertY = kwargs['hamCfg']['invertY']
        self.SpacingX = self.L / kwargs['motorCfg']['slots'] / kwargs['canvasCfg']['pixDiv'][0]
        self.SpacingY = self.H / kwargs['canvasCfg']['pixDiv'][1]
        self.Cspacing = kwargs['canvasCfg']['canvasSpacing'] / self.H
        self.hmRegions = kwargs['hamCfg']['hmRegions']
        self.mecRegions = kwargs['hamCfg']['mecRegions']
        self.allMecRegions = not self.hmRegions
        combinedList = list(self.hmRegions.items()) + list(self.mecRegions.items())
        self.allRegions = dict(sorted(combinedList, key=lambda x: x[0]))

        if self.allMecRegions:
            self.hmRegionsIndex = np.zeros(len(self.hmRegions), dtype=np.int32)
            self.mecRegionsIndex = np.zeros(len(self.mecRegions) + 1, dtype=np.int32)
        else:
            self.hmRegionsIndex = np.zeros(len(self.hmRegions) + 1, dtype=np.int32)
            self.mecRegionsIndex = np.zeros(len(self.mecRegions), dtype=np.int32)
        self.lenUnknowns = 0

        self.yIndexesVacLower, self.yIndexesVacUpper = [], []
        self.yIndexesBackIron, self.yIndexesBladeRotor, self.yIndexesAirGap = [], [], []
        self.yIndexesLowerSlot, self.yIndexesUpperSlot, self.yIndexesYoke = [], [], []

        self.removeLowerCoilIdxs, self.removeUpperCoilIdxs = [], []

        self.yListPixelsPerRegion = []
        self.yIndexesHM, self.yIndexesMEC = [], []
        self.xBoundaryList, self.yBoundaryList = [], []

        # X-direction
        self.ppSlotpitch = self.setPixelsPerLength(length=self.slotpitch, minimum=2, direction="x")
        self.ppAirBuffer = self.setPixelsPerLength(length=self.Airbuffer, minimum=1, direction="x")
        self.ppTooth = self.setPixelsPerLength(length=self.wt, minimum=1, direction="x")
        self.ppSlot = self.ppSlotpitch - self.ppTooth
        self.ppEndTooth = self.setPixelsPerLength(length=self.endTooth, minimum=1, direction="x")
        # Y-direction
        self.ppVac = self.setPixelsPerLength(length=self.vac, override=1, direction="y")
        self.ppYoke = self.setPixelsPerLength(length=self.hy, minimum=1, direction="y")
        self.ppAirGap = self.setPixelsPerLength(length=self.g, minimum=1, direction="y")
        self.ppBladeRotor = self.setPixelsPerLength(length=self.dr, minimum=1, direction="y")
        self.ppBackIron = self.setPixelsPerLength(length=self.bi, minimum=1, direction="y")
        self.ppSlotHeight = self.setPixelsPerLength(length=self.hs, minimum=2, direction="y")

        # Keeping the airgap odd allows for a center airgap thrust calculation
        if self.ppAirGap % 2 == 0:
            self.ppAirGap += 1

        if self.ppSlotHeight % 2 != 0:
            self.ppSlotHeight += 1

        # Determine pixels per harmonic region
        self.ppHM = 0
        for each in self.hmRegions.values():
            if each.split('_')[0] != 'vac':
                self.ppHM += self.__dict__[self.getFullRegionDict()[each]['pp']]
        self.ppMEC = 0
        for core in self.mecRegions.values():
            for pixel in self.getFullRegionDict()[core]['pp'].split(', '):
                self.ppMEC += self.ppSlotHeight // 2 if pixel == PP_SLOTHEIGHT else self.__dict__[pixel]

        self.ppH = self.ppMEC + self.ppHM
        self.ppL = (self.slots - 1) * self.ppSlotpitch + self.ppSlot + 2 * self.ppEndTooth + 2 * self.ppAirBuffer
        self.matrix = np.array([[type('', (Node,), {}) for _ in range(self.ppL)] for _ in range(self.ppMEC + self.ppHM)])

        self.toothArray, self.slotArray, self.bufferArray = np.zeros((3, 1), dtype=np.int16)

        self.mecRegionLength = self.ppMEC * self.ppL
        self.modelHeight = 0
        for region, values in self.getFullRegionDict().items():
            if region.split('_')[0] != 'vac':
                for spatial in values['spatial'].split(', '):
                    self.modelHeight += self.__dict__[spatial] / 2 if spatial == 'hs' else self.__dict__[spatial]

        # Update the hm and mec RegionIndex
        self.setRegionIndices()

        self.xListSpatialDatum = [self.Airbuffer, self.endTooth] + [self.ws, self.wt] * (self.slots - 1) + [self.ws, self.endTooth, self.Airbuffer]
        self.xListPixelsPerRegion = [self.ppAirBuffer, self.ppEndTooth] + [self.ppSlot, self.ppTooth] * (self.slots - 1) + [self.ppSlot, self.ppEndTooth, self.ppAirBuffer]
        self.xMeshSizes = np.zeros(len(self.xListSpatialDatum), dtype=np.float64)
        for Cnt in range(len(self.xMeshSizes)):
            self.xMeshSizes[Cnt] = meshBoundary(self.xListSpatialDatum[Cnt], self.xListPixelsPerRegion[Cnt])
            if self.xMeshSizes[Cnt] < 0:
                print('negative x mesh sizes', Cnt)
                return

        self.yMeshSizes = []
        offsetList = []
        cnt, offsetLower, offsetUpper = 0, 0, 0
        for region in self.getFullRegionDict():
            if region.split('_')[0] != 'vac':
                # Set up the y-axis indexes
                for pp in self.getFullRegionDict()[region]['pp'].split(', '):
                    offsetLower = offsetUpper
                    offsetUpper += self.__dict__[pp] // 2 if pp == PP_SLOTHEIGHT else self.__dict__[pp]
                    offsetList.append((offsetLower, offsetUpper))
                for inner_cnt, spatial in enumerate(self.getFullRegionDict()[region]['spatial'].split(', ')):
                    pixelKey = self.getFullRegionDict()[region]['pp'].split(', ')[inner_cnt]
                    pixelVal = self.__dict__[pixelKey] // 2 if pixelKey == PP_SLOTHEIGHT else self.__dict__[pixelKey]
                    spatialVal = self.__dict__[spatial] / 2 if pixelKey == PP_SLOTHEIGHT else self.__dict__[spatial]
                    self.yListPixelsPerRegion.append(pixelVal)
                    self.yMeshSizes.append(meshBoundary(spatialVal, pixelVal))
                    cnt += 1

        # Initialize the y-axis index attributes format: self.yIndexes___
        innerCnt = 0
        for region in self.getFullRegionDict():
            if region.split('_')[0] != 'vac':
                for idx in self.getFullRegionDict()[region]['idx'].split(', '):
                    self.__dict__[idx] = list(range(offsetList[innerCnt][0], offsetList[innerCnt][1]))
                    if region != 'vac_upper':
                        self.yBoundaryList.append(self.__dict__[idx][-1])
                    if region in self.hmRegions.values():
                        self.yIndexesHM.extend(self.__dict__[idx])
                    elif region in self.mecRegions.values():
                        self.yIndexesMEC.extend(self.__dict__[idx])
                    innerCnt += 1

        self.yIdxCenterAirGap = self.yIndexesAirGap[0] + self.ppAirGap // 2

        # Scaling values for MMF-source distribution in section 2.2, equation 18, figure 5
        fraction = 0.5
        doubleBias = self.ppSlotHeight - fraction
        self.doubleCoilScaling = np.arange(fraction, doubleBias + fraction, 1)
        if self.invertY:
            self.doubleCoilScaling = np.flip(self.doubleCoilScaling)

        # Thrust of the entire integration region
        self.Fx = 0.0
        self.Fy = 0.0

    def buildGrid(self):
        # TODO The best way to do this is to make the class work for integral mainly and then create a new instance of the class which has double layer
        #  I need to be careful not to mess up my basline motor so I should keep only integral windings for platypus. Start tracing variables like removed slots ...

        #  Initialize grid with Nodes
        listOffset = self.ppAirBuffer + self.ppEndTooth
        oldTooth, newTooth, oldSlot, newSlot, slotCount = 0, listOffset, 0, 0, 0
        fullToothArray = []
        for slotCount in range(self.slots - 1):
            oldSlot = newTooth
            newSlot = oldSlot + self.ppSlot
            oldTooth = newSlot
            newTooth = oldTooth + self.ppTooth
            fullToothArray += list(range(oldTooth, newTooth))

        leftEndTooth = list(range(self.ppAirBuffer, listOffset))
        rightEndTooth = list(range(fullToothArray[-1] + self.ppSlot + 1, fullToothArray[-1] + self.ppSlot + self.ppEndTooth + 1))
        self.toothArray = leftEndTooth + fullToothArray + rightEndTooth
        self.slotArray = [x for x in range(self.ppAirBuffer, self.ppL - self.ppAirBuffer) if x not in self.toothArray]
        self.bufferArray = [x for x in list(range(self.ppL)) if x not in self.slotArray and x not in self.toothArray]

        # Split slots into respective phases
        offset = self.ppSlot*math.ceil(self.q)

        def consecutiveCount(numbers):
            _idx = 0
            while _idx < (len(numbers) - 2) and (numbers[_idx+1] - numbers[_idx] == 1):
                _idx += 1
            return _idx + 1

        if len(self.removeUpperCoils) == 0:
            upperSlotArray = self.slotArray
        else:
            leftRemoveUpper = consecutiveCount(self.removeUpperCoils)
            upperSlotArray = self.slotArray[leftRemoveUpper*offset:-(len(self.removeUpperCoils)-leftRemoveUpper)*offset]
        if len(self.removeLowerCoils) == 0:
            lowerSlotArray = self.slotArray
        else:
            leftRemoveLower = consecutiveCount(self.removeLowerCoils)
            lowerSlotArray = self.slotArray[leftRemoveLower*offset:-(len(self.removeLowerCoils)-leftRemoveLower)*offset]

        upper_slotArrayA, upper_slotArrayB, upper_slotArrayC = [], [], []
        lower_slotArrayA, lower_slotArrayB, lower_slotArrayC = [], [], []
        for threeSlots in range(0, self.slots, 3):
            upper_slotArrayA += upperSlotArray[threeSlots*offset:(threeSlots+1)*offset]
            upper_slotArrayB += upperSlotArray[(threeSlots+2)*offset:(threeSlots+3)*offset]
            upper_slotArrayC += upperSlotArray[(threeSlots+1)*offset:(threeSlots+2)*offset]
            lower_slotArrayA += lowerSlotArray[threeSlots*offset:(threeSlots+1)*offset]
            lower_slotArrayB += lowerSlotArray[(threeSlots+2)*offset:(threeSlots+3)*offset]
            lower_slotArrayC += lowerSlotArray[(threeSlots+1)*offset:(threeSlots+2)*offset]

        self.upper_slotsA = np.array(upper_slotArrayA)
        self.lower_slotsA = np.array(lower_slotArrayA)
        self.upper_slotsB = np.array(upper_slotArrayB)
        self.lower_slotsB = np.array(lower_slotArrayB)
        self.upper_slotsC = np.array(upper_slotArrayC)
        self.lower_slotsC = np.array(lower_slotArrayC)

        # Sort coils into direction of current (ex, in and out of page)
        def phaseToInOut(_ppSlot, _array, _offset):
            storeArray = np.empty(0, dtype=int)
            for phase in np.array_split(_array, len(_array) // _ppSlot)[_offset::2]:
                storeArray = np.concatenate((storeArray, phase))
            return storeArray

        # TODO I need to make this winding shifting more robust. I need to find out how much I should shift by and then look into plotting multiple aigrap plots of stator sine wave
        if not all(list(map(lambda x: len(x) == len(self.upper_slotsA), [self.lower_slotsA, self.upper_slotsB, self.lower_slotsB, self.upper_slotsC, self.lower_slotsC]))):
            self.writeErrorToDict(key='name',
                                  error=Error.buildFromScratch(name='windingError',
                                                               description='Validate that there are no monopoles and that each phase has the same number of terminals\n' +
                                                                           f'phases terminals: {list(map(lambda x: len(x), [self.upper_slotsA, self.lower_slotsA, self.upper_slotsB, self.lower_slotsB, self.upper_slotsC, self.lower_slotsC]))}',
                                                               cause=True))

        # The first coil of phases A and B are positive
        self.inUpper_slotsA = phaseToInOut(self.ppSlot, self.upper_slotsA, 1)
        self.outUpper_slotsA = phaseToInOut(self.ppSlot, self.upper_slotsA, 0)
        self.inLower_slotsA = phaseToInOut(self.ppSlot, self.lower_slotsA, 1)
        self.outLower_slotsA = phaseToInOut(self.ppSlot, self.lower_slotsA, 0)
        self.inUpper_slotsB = phaseToInOut(self.ppSlot, self.upper_slotsB, 1)
        self.outUpper_slotsB = phaseToInOut(self.ppSlot, self.upper_slotsB, 0)
        self.inLower_slotsB = phaseToInOut(self.ppSlot, self.lower_slotsB, 1)
        self.outLower_slotsB = phaseToInOut(self.ppSlot, self.lower_slotsB, 0)
        # The first coil of phase C is negative
        self.inUpper_slotsC = phaseToInOut(self.ppSlot, self.upper_slotsC, 0)
        self.outUpper_slotsC = phaseToInOut(self.ppSlot, self.upper_slotsC, 1)
        self.inLower_slotsC = phaseToInOut(self.ppSlot, self.lower_slotsC, 0)
        self.outLower_slotsC = phaseToInOut(self.ppSlot, self.lower_slotsC, 1)

        for idx in self.removeUpperCoils:
            coilOffset = idx*self.ppSlot
            self.removeUpperCoilIdxs += self.slotArray[coilOffset:coilOffset + self.ppSlot]

        for idx in self.removeLowerCoils:
            coilOffset = idx*self.ppSlot
            self.removeLowerCoilIdxs += self.slotArray[coilOffset:coilOffset + self.ppSlot]

        self.xBoundaryList = [self.bufferArray[self.ppAirBuffer - 1],
                              self.toothArray[self.ppEndTooth - 1]] + self.slotArray[self.ppSlot - 1::self.ppSlot] \
                             + self.toothArray[self.ppEndTooth + self.ppTooth - 1:-self.ppEndTooth:self.ppTooth] + [self.toothArray[-1],
                                                                                                                    self.bufferArray[-1]]

        a, b = 0, 0
        c, d = 0, 0
        yCnt = 0
        # Assign spatial data to the nodes
        while a < self.ppH:
            xCnt = 0
            delY = self.yMeshSizes[c]
            while b < self.ppL:
                delX = self.xMeshSizes[d]
                self.matrix[a][b] = Node.buildFromScratch(iIndex=[b, a], iXinfo=[xCnt, delX], iYinfo=[yCnt, delY], model=self)
                # Keep track of the x coordinate for each node
                xCnt += delX
                if b in self.xBoundaryList:
                    d += 1
                b += 1
            d = 0
            # Keep track of the y coordinate for each node
            yCnt += delY
            # TODO We cannot have the last row in this list so it must be unwritten
            if a in self.yBoundaryList:
                c += 1
            b = 0
            a += 1

    def setPixelsPerLength(self, length, minimum=1, override=0, direction=None):

        if direction == "x":
            pixels = round(length / self.SpacingX)
        elif direction == "y":
            pixels = round(length / self.SpacingY)
        else:
            print("Direction parameter options: x, y")
            return

        if override >= 1:
            return override
        else:
            return minimum if pixels < minimum else pixels

    def setRegionIndices(self):

        # Create the starting index for each region in the columns of matrix A
        regionIndex, hmCount, mecCount = 0, 0, 0
        for index, name in self.allRegions.items():
            if index in self.hmRegions:
                # Dirichlet boundaries have half the unknown coefficients
                if self.hmRegions[index].split('_')[0] == 'vac':
                    self.hmRegionsIndex[hmCount] = regionIndex
                    regionIndex += len(self.n)
                else:
                    self.hmRegionsIndex[hmCount] = regionIndex
                    regionIndex += 2 * len(self.n)
                hmCount += 1

                if index == list(self.hmRegions)[-1]:
                    self.hmRegionsIndex[-1] = regionIndex

        # TODO the for loop excluded += self.mecRegionLength
            if index in self.mecRegions:
                self.mecRegionsIndex[mecCount] = regionIndex
                regionIndex += self.mecRegionLength
                mecCount += 1

                if index == list(self.mecRegions)[-1] and self.allMecRegions:
                    self.mecRegionsIndex[-1] = regionIndex

    def getFullRegionDict(self):
        config = configparser.ConfigParser()
        config.read('Properties.ini')
        regDict = {}

        for index, name in self.allRegions.items():
            regDict[name] = {'pp': config.get('PP', name),
                               'bc': config.get('BC', name),
                               'idx': config.get('Y_IDX', name),
                               'spatial': config.get('SPATIAL', name)}

            # We need to reverse the Properties.ini file
            if self.invertY and index in self.mecRegions:
                for each in regDict[name]:
                    stringList = ''
                    temp = regDict[name][each].split(', ')
                    temp.reverse()
                    for cnt, string in enumerate(temp):
                        if cnt == 0:
                            stringList += string
                        else:
                            stringList += f', {string}'
                    regDict[name][each] = stringList

        return regDict

    def getLastAndNextRegionName(self, name):
        if list(self.getFullRegionDict()).index(name) == 0:
            previous = None
        else:
            previous = list(self.getFullRegionDict())[list(self.getFullRegionDict()).index(name)-1]
        if list(self.getFullRegionDict()).index(name) == len(self.getFullRegionDict()) - 1:
            next_ = None
        else:
            next_ = list(self.getFullRegionDict())[list(self.getFullRegionDict()).index(name)+1]

        return previous, next_

    # This function is written to catch any errors in the mapping between canvas and space for both x and y coordinates
    def checkSpatialMapping(self):

        spatialDomainFlag = False

        # Define Indexes
        idxLeftAirBuffer = 0
        idxLeftEndTooth = idxLeftAirBuffer + self.ppAirBuffer
        idxSlot = idxLeftEndTooth + self.ppEndTooth
        idxTooth = idxSlot + self.ppSlot
        idxRightEndTooth = self.ppL - self.ppAirBuffer - self.ppEndTooth
        idxRightAirBuffer = idxRightEndTooth + self.ppEndTooth

        # The difference between expected and actual is rounded due to quantization error
        #  in the 64 bit floating point arithmetic

        # X direction Checks

        yIdx = 0
        # Check slotpitch
        if round(self.slotpitch - (self.matrix[yIdx][idxTooth + self.ppTooth].x - self.matrix[yIdx][idxSlot].x), 12) != 0:
            print(f'flag - slotpitch: {self.slotpitch - (self.matrix[yIdx][idxTooth + self.ppTooth].x - self.matrix[yIdx][idxSlot].x)}')
            spatialDomainFlag = True
        # Check slot width
        if round(self.ws - (self.matrix[yIdx][idxTooth].x - self.matrix[yIdx][idxSlot].x), 12) != 0:
            print(f'flag - slots: {self.ws - (self.matrix[yIdx][idxTooth].x - self.matrix[yIdx][idxSlot].x)}')
            spatialDomainFlag = True
        # Check tooth width
        if round(self.wt - (self.matrix[yIdx][idxTooth + self.ppTooth].x - self.matrix[yIdx][idxTooth].x), 12) != 0:
            print(f'flag - teeth: {self.wt - (self.matrix[yIdx][idxTooth + self.ppTooth].x - self.matrix[yIdx][idxTooth].x)}')
            spatialDomainFlag = True
        # Check left end tooth
        if round(self.endTooth - (self.matrix[yIdx][idxSlot].x - self.matrix[yIdx][idxLeftEndTooth].x), 12) != 0:
            print(f'flag - left end tooth: {self.endTooth - (self.matrix[yIdx][idxSlot].x - self.matrix[yIdx][idxLeftEndTooth].x)}')
            spatialDomainFlag = True
        # Check right end tooth
        if round(self.endTooth - (self.matrix[yIdx][idxRightAirBuffer].x - self.matrix[yIdx][idxRightEndTooth].x), 12) != 0:
            print(f'flag - right end tooth: {self.endTooth - (self.matrix[yIdx][idxRightAirBuffer].x - self.matrix[yIdx][idxRightEndTooth].x)}')
            spatialDomainFlag = True
        # Check left air buffer
        if round(self.Airbuffer - (self.matrix[yIdx][idxLeftEndTooth].x - self.matrix[yIdx][idxLeftAirBuffer].x), 12) != 0:
            print(f'flag - left air buffer: {self.Airbuffer - (self.matrix[yIdx][idxLeftEndTooth].x - self.matrix[yIdx][idxLeftAirBuffer].x)}')
            spatialDomainFlag = True
        # Check right air buffer
        if round(self.Airbuffer - (self.matrix[yIdx][-1].x + self.matrix[yIdx][-1].lx - self.matrix[yIdx][idxRightAirBuffer].x), 12) != 0:
            print(f'flag - right air buffer: {self.Airbuffer - (self.matrix[yIdx][-1].x + self.matrix[yIdx][-1].lx - self.matrix[yIdx][idxRightAirBuffer].x)}')
            spatialDomainFlag = True

        # Y direction Checks

        def yUpperPos(_list):
            return self.matrix[_list[-1]][xIdx].y + self.matrix[_list[-1]][xIdx].ly

        xIdx = 0
        # Check Blade Rotor
        diffBladeRotorDims = self.dr - (yUpperPos(self.yIndexesBladeRotor) - self.matrix[self.yIndexesBladeRotor[0]][xIdx].y)
        if round(diffBladeRotorDims, 12) != 0:
            print(f'flag - blade rotor: {diffBladeRotorDims}')
            spatialDomainFlag = True
        # Check Air Gap
        diffAirGapDims = self.g - (yUpperPos(self.yIndexesAirGap) - self.matrix[self.yIndexesAirGap[0]][xIdx].y)
        if round(diffAirGapDims, 12) != 0:
            print(f'flag - air gap: {diffAirGapDims}')
            spatialDomainFlag = True
        # Check Lower Slot Height
        diffLowerSlotHeightDims = self.hs/2 - (yUpperPos(self.yIndexesLowerSlot) - self.matrix[self.yIndexesLowerSlot[0]][xIdx].y)
        if round(diffLowerSlotHeightDims, 12) != 0:
            print(f'flag - slot height: {diffLowerSlotHeightDims}')
            spatialDomainFlag = True
        # Check Upper Slot Height
        diffUpperSlotHeightDims = self.hs/2 - (yUpperPos(self.yIndexesUpperSlot) - self.matrix[self.yIndexesUpperSlot[0]][xIdx].y)
        if round(diffUpperSlotHeightDims, 12) != 0:
            print(f'flag - slot height: {diffUpperSlotHeightDims}')
            spatialDomainFlag = True
        # Check Yoke
        diffYokeDims = self.hy - (yUpperPos(self.yIndexesYoke) - self.matrix[self.yIndexesYoke[0]][xIdx].y)
        if round(diffYokeDims, 12) != 0:
            print(f'flag - yoke: {diffYokeDims}')
            spatialDomainFlag = True
        # Check Back Iron
        diffBackIronDims = self.bi - (yUpperPos(self.yIndexesBackIron) - self.matrix[self.yIndexesBackIron[0]][xIdx].y)
        if round(diffBackIronDims, 12) != 0:
            print(f'flag - back iron: {diffBackIronDims}')
            spatialDomainFlag = True

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='domainMapping',
                                                           description='ERROR - The spatial domain does not match with the canvas domain',
                                                           cause=spatialDomainFlag))

    def timePlex(self):
        return cmath.exp(j_plex * 2 * pi * self.f * self.t)

    def angleA(self):
        return cmath.exp(0)

    def angleB(self):
        # TODO This could be an error since rotation is generally: A(0°), B(120°), C(240°)
        return cmath.exp(-j_plex * pi * 2 / 3)

    def angleC(self):
        return cmath.exp(j_plex * pi * 2 / 3)

class Node(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for attr_key in kwargs:
                if type(kwargs[attr_key]) == list and kwargs[attr_key][0] == 'plex_Signature':
                    self.__dict__[attr_key] = rebuildPlex(kwargs[attr_key])
                else:
                    self.__dict__[attr_key] = kwargs[attr_key]
            return

        model = kwargs['model']

        self.xIndex = kwargs['iIndex'][0]
        self.yIndex = kwargs['iIndex'][1]

        # x-direction
        self.x = kwargs['iXinfo'][0]
        self.lx = kwargs['iXinfo'][1]

        # y-direction
        self.y = kwargs['iYinfo'][0]
        self.ly = kwargs['iYinfo'][1]

        # Cross sectional area
        self.Szy = self.ly * model.D
        self.Sxz = self.lx * model.D

        self.xCenter = self.x + self.lx / 2
        self.yCenter = self.y + self.ly / 2

        # Node properties
        if self.yIndex in model.yIndexesVacLower or self.yIndex in model.yIndexesVacUpper:
            self.material = 'vacuum'
            self.ur = model.air.ur
            self.sigma = model.air.sigma
        elif self.yIndex in model.yIndexesYoke and self.xIndex not in model.bufferArray:
            self.material = 'iron'
            self.ur = model.iron.ur
            self.sigma = model.iron.sigma
        elif self.yIndex in model.yIndexesAirGap:
            self.material = 'vacuum'
            self.ur = model.air.ur
            self.sigma = model.air.sigma
        elif self.yIndex in model.yIndexesBladeRotor:
            self.material = 'aluminum'
            self.ur = model.alum.ur
            self.sigma = model.alum.sigma
        elif self.yIndex in model.yIndexesBackIron:
            self.material = 'iron'
            self.ur = model.iron.ur
            self.sigma = model.iron.sigma
        else:
            if self.yIndex in model.yIndexesUpperSlot:
                aIdx = model.upper_slotsA
                bIdx = model.upper_slotsB
                cIdx = model.upper_slotsC
            elif self.yIndex in model.yIndexesLowerSlot:
                aIdx = model.lower_slotsA
                bIdx = model.lower_slotsB
                cIdx = model.lower_slotsC
            else:
                aIdx = []
                bIdx = []
                cIdx = []

            if self.xIndex in model.toothArray:
                self.material = 'iron'
                self.ur = model.iron.ur
                self.sigma = model.iron.sigma
            elif self.xIndex in model.bufferArray:
                self.material = 'vacuum'
                self.ur = model.air.ur
                self.sigma = model.air.sigma
            elif self.xIndex in aIdx:
                self.material = 'copperA'
                self.ur = model.copper.ur
                self.sigma = model.copper.sigma
            elif self.xIndex in bIdx:
                self.material = 'copperB'
                self.ur = model.copper.ur
                self.sigma = model.copper.sigma
            elif self.xIndex in cIdx:
                self.material = 'copperC'
                self.ur = model.copper.ur
                self.sigma = model.copper.sigma
            elif self.xIndex in model.removeLowerCoilIdxs + model.removeUpperCoilIdxs:
                self.material = 'vacuum'
                self.ur = model.air.ur
                self.sigma = model.air.sigma
            else:
                self.material = ''

        # Define colour
        idx = np.where(matList == self.material)
        if len(idx[0] == 1):
            matIdx = idx[0][0]
            self.colour = matList[matIdx][1]
        else:
            self.colour = 'orange'

        self.Rx, self.Ry = self.getReluctance(model)

        scalingLower, scalingUpper = 0.0, 0.0
        isCurrentCu = self.material[:-1] == 'copper'
        if self.yIndex in model.yIndexesLowerSlot and self.xIndex in model.slotArray:
            if self.xIndex in model.lower_slotsA:
                angle_plex = model.angleA()
            elif self.xIndex in model.lower_slotsB:
                angle_plex = model.angleB()
            elif self.xIndex in model.lower_slotsC:
                angle_plex = model.angleC()
            else:
                angle_plex = 0.0

            # Set the scaling factor for MMF in equation 18
            if isCurrentCu:
                index_ = model.yIndexesLowerSlot.index(self.yIndex)
                if model.invertY:
                    index_ += len(model.doubleCoilScaling) // 2
                scalingLower = model.doubleCoilScaling[index_]
            else:
                scalingLower = 0.0

            # Determine coil terminal direction
            if self.xIndex in model.inLower_slotsA or self.xIndex in model.inLower_slotsB or self.xIndex in model.inLower_slotsC:
                inOutCoeffMMF = -1
            elif self.xIndex in model.outLower_slotsA or self.xIndex in model.outLower_slotsB or self.xIndex in model.outLower_slotsC:
                inOutCoeffMMF = 1
            else:
                inOutCoeffMMF = 0

        elif self.yIndex in model.yIndexesUpperSlot and self.xIndex in model.slotArray:
            if self.xIndex in model.upper_slotsA:
                angle_plex = model.angleA()
            elif self.xIndex in model.upper_slotsB:
                angle_plex = model.angleB()
            elif self.xIndex in model.upper_slotsC:
                angle_plex = model.angleC()
            else:
                angle_plex = 0.0

            # Set the scaling factor for MMF in equation 18
            yLowerCoilIdx = self.yIndex + model.ppSlotHeight // 2 if model.invertY else self.yIndex - model.ppSlotHeight // 2
            isLowerCu = model.matrix[yLowerCoilIdx][self.xIndex].material[:-1] == 'copper'
            # 2 coils in slot
            if isCurrentCu and isLowerCu:
                index_ = model.yIndexesUpperSlot.index(self.yIndex)
                if not model.invertY:
                    index_ += len(model.doubleCoilScaling) // 2
                scalingUpper = model.doubleCoilScaling[index_]
            # coil in upper slot only
            elif isCurrentCu and not isLowerCu:
                index_ = model.yIndexesUpperSlot.index(self.yIndex)
                if model.invertY:
                    index_ -= len(model.doubleCoilScaling) // 2
                scalingUpper = model.doubleCoilScaling[index_]
            else:
                scalingUpper = 0.0

            # Determine coil terminal direction
            if self.xIndex in model.inUpper_slotsA or self.xIndex in model.inUpper_slotsB or self.xIndex in model.inUpper_slotsC:
                inOutCoeffMMF = -1
            elif self.xIndex in model.outUpper_slotsA or self.xIndex in model.outUpper_slotsB or self.xIndex in model.outUpper_slotsC:
                inOutCoeffMMF = 1
            else:
                inOutCoeffMMF = 0

        else:
            angle_plex = 0.0
            scalingLower = 0.0
            scalingUpper = 0.0
            inOutCoeffMMF = 0

        # Assign the current and MMF per node
        self.Iph = model.Ip * angle_plex * model.timePlex()
        self.MMF = inOutCoeffMMF * model.N * self.Iph / (2 * model.ppSlot)
        if self.yIndex in model.yIndexesLowerSlot and self.xIndex in model.slotArray:
            self.MMF *= scalingLower
        elif self.yIndex in model.yIndexesUpperSlot and self.xIndex in model.slotArray:
            self.MMF *= scalingUpper
        else:
            self.MMF = 0.0

        # Potential
        self.Yk = np.cdouble(0.0)
        # B field
        self.Bx, self.By, self.B = np.zeros(3, dtype=np.cdouble)
        # Flux
        self.phiXp, self.phiXn, self.phiYp, self.phiYn, self.phiError = np.zeros(5, dtype=np.cdouble)

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    def __eq__(self, otherObject):
        if not isinstance(otherObject, Node):
            # don't attempt to compare against unrelated types
            return NotImplemented
        # If the objects are the same then set the IDs to be equal
        elif self.__dict__.items() == otherObject.__dict__.items():
            for attr, val in otherObject.__dict__.items():
                return self.__dict__[attr] == otherObject.__dict__[attr]
        # The objects are not the same
        else:
            pass

    def drawNode(self, canvasSpacing, overRideColour, c, nodeWidth, outline='black'):

        x_old = self.x*canvasSpacing
        x_new = x_old + self.lx*canvasSpacing
        y_old = self.y*canvasSpacing
        y_new = y_old + self.ly*canvasSpacing

        if overRideColour:
            fillColour = overRideColour
        else:
            idx = np.where(matList == self.material)
            if len(idx[0] == 1):
                matIdx = idx[0][0]
                self.colour = matList[matIdx][1]
            else:
                self.colour = 'orange'
            fillColour = self.colour

        c.create_rectangle(x_old, y_old, x_new, y_new, width=nodeWidth, fill=fillColour, outline=outline)

    def getReluctance(self, model, isVac=False):

        ResX = self.lx / (2 * uo * self.ur * self.Szy)
        if isVac:  # Create a fake vac node
            vacHeight = model.vac / model.ppVac
            ResY = vacHeight / (2 * uo * model.air.ur * self.Sxz)
        else:
            ResY = self.ly / (2 * uo * self.ur * self.Sxz)

        return ResX, ResY


class Region(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for key in kwargs:
                if key in ['an', 'bn']:
                    if type(kwargs[key]) == np.ndarray:
                        valList = [rebuildPlex(index) for index in kwargs[key]]
                        valArray = np.array(valList)
                        self.__dict__[key] = valArray
                    else:
                        self.__dict__[key] = kwargs[key]
                else:
                    self.__dict__[key] = kwargs[key]
            return

        self.type = kwargs['type']
        self.an = kwargs['an']
        self.bn = kwargs['bn']

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def rebuildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)


def meshBoundary(spatial, pixels):

    return spatial / pixels