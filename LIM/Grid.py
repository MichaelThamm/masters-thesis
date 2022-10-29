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
        xMeshIndexes = kwargs['canvasCfg']['xMeshIndexes']
        yMeshIndexes = kwargs['canvasCfg']['yMeshIndexes']
        self.SpacingX = self.L / kwargs['motorCfg']['slots'] / kwargs['canvasCfg']['pixDiv'][0]
        self.SpacingY = self.H / kwargs['canvasCfg']['pixDiv'][1]
        self.Cspacing = kwargs['canvasCfg']['canvasSpacing'] / self.H
        self.meshDensity = kwargs['canvasCfg']['meshDensity']
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
        self.xFirstEdgeNodes, self.xSecondEdgeNodes = [], []
        self.yFirstEdgeNodes, self.ySecondEdgeNodes = [], []

        self.yListPixelsPerRegion = []
        self.yIndexesHM, self.yIndexesMEC = [], []
        self.xBoundaryList, self.yBoundaryList = [], []

        self.yMeshSizes = []

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
                self.ppMEC += self.ppSlotHeight // 2 if pixel == 'ppSlotHeight' else self.__dict__[pixel]

        self.ppH = self.ppMEC + self.ppHM
        self.ppL = (self.slots - 1) * self.ppSlotpitch + self.ppSlot + 2 * self.ppEndTooth + 2 * self.ppAirBuffer
        self.matrix = np.array([[type('', (Node,), {}) for _ in range(self.ppL)] for _ in range(self.ppMEC + self.ppHM)])

        self.toothArray, self.slotArray, self.bufferArray = np.zeros((3, 1), dtype=np.int16)

        self.xMeshSizes = np.zeros(len(xMeshIndexes), dtype=np.float64)

        # Mesh sizing
        self.fractionSize = (1 / self.meshDensity[0] + 1 / self.meshDensity[1])

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
        for Cnt in range(len(self.xMeshSizes)):
            self.xMeshSizes[Cnt] = meshBoundary(self.xListSpatialDatum[Cnt], self.xListPixelsPerRegion[Cnt], self.SpacingX, self.fractionSize, sum(xMeshIndexes[Cnt]), self.meshDensity)
            if self.xMeshSizes[Cnt] < 0:
                print('negative x mesh sizes', Cnt)
                return

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
                    self.yMeshSizes.append(meshBoundary(spatialVal, pixelVal, self.SpacingY, self.fractionSize,
                                                        sum(yMeshIndexes[cnt]), self.meshDensity))
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

        # Thrust of the entire integration region
        self.Fx = 0.0
        self.Fy = 0.0

    def buildGrid(self, xMeshIndexes, yMeshIndexes):
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
        # TODO Is this good math?
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
        def phaseToInOut(_array, _offset):
            storeArray = np.empty(0, dtype=int)
            for phase in np.array_split(_array, len(_array) // 2)[_offset::2]:
                storeArray = np.concatenate((storeArray, phase))
            return storeArray

        # TODO I need to make this winding shifting more robust. I need to find out how much I should shift by and then look into plotting multiple aigrap plots of stator sine wave
        if not all(list(map(lambda x: len(x) == len(self.upper_slotsA), [self.lower_slotsA, self.upper_slotsB, self.lower_slotsB, self.upper_slotsC, self.lower_slotsC]))):
            self.writeErrorToDict(key='name',
                                  error=Error.buildFromScratch(name='windingError',
                                                               description='Validate that there are no monopoles and that each phase has the same number of terminals\n' +
                                                                           f'phases terminals: {list(map(lambda x: len(x), [self.upper_slotsA, self.lower_slotsA, self.upper_slotsB, self.lower_slotsB, self.upper_slotsC, self.lower_slotsC]))}',
                                                               cause=True))

        self.inUpper_slotsA = phaseToInOut(self.upper_slotsA, 1)
        self.outUpper_slotsA = phaseToInOut(self.upper_slotsA, 0)
        self.inLower_slotsA = phaseToInOut(self.lower_slotsA, 1)
        self.outLower_slotsA = phaseToInOut(self.lower_slotsA, 0)
        self.inUpper_slotsB = phaseToInOut(self.upper_slotsB, 1)
        self.outUpper_slotsB = phaseToInOut(self.upper_slotsB, 0)
        self.inLower_slotsB = phaseToInOut(self.lower_slotsB, 1)
        self.outLower_slotsB = phaseToInOut(self.lower_slotsB, 0)
        # The first coil of phase C is negative unlike phases A and B
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

        # MeshIndexes is a list of 1s and 0s for each boundary for each region to say whether or not
        #  dense meshing is required at the boundary
        # X Mesh Density
        Cnt = 0
        idxOffset = self.xListPixelsPerRegion[Cnt]
        idxLeft, idxRight = 0, idxOffset
        idxList = range(self.ppL)
        for boundary in xMeshIndexes:
            if boundary[0]:  # Left Boundary in the region
                firstIndexes = idxList[idxLeft]
                secondIndexes = idxList[idxLeft + 1]
                self.xFirstEdgeNodes.append(firstIndexes)
                self.xSecondEdgeNodes.append(secondIndexes)
            if boundary[1]:  # Right Boundary in the region
                firstIndexes = idxList[idxRight - 1]
                secondIndexes = idxList[idxRight - 2]
                self.xFirstEdgeNodes.append(firstIndexes)
                self.xSecondEdgeNodes.append(secondIndexes)
            idxLeft += idxOffset
            if Cnt < len(self.xListPixelsPerRegion) - 1:
                idxOffset = self.xListPixelsPerRegion[Cnt+1]
            idxRight += idxOffset
            Cnt += 1

        # Y Mesh Density
        Cnt = 0
        idxOffset = self.yListPixelsPerRegion[Cnt]
        idxLeft, idxRight = 0, idxOffset
        # TODO Here - yMeshIndexes is not robust since removal of an and bn prior
        idxList = range(self.ppH)
        for boundary in yMeshIndexes:
            if boundary[0]:  # Left Boundary in the region
                firstIndexes = idxList[idxLeft]
                secondIndexes = idxList[idxLeft + 1]
                self.yFirstEdgeNodes.append(firstIndexes)
                self.ySecondEdgeNodes.append(secondIndexes)
            if boundary[1]:  # Right Boundary in the region
                firstIndexes = idxList[idxRight - 1]
                secondIndexes = idxList[idxRight - 2]
                self.yFirstEdgeNodes.append(firstIndexes)
                self.ySecondEdgeNodes.append(secondIndexes)
            idxLeft += idxOffset
            if Cnt < len(self.yListPixelsPerRegion) - 1:
                idxOffset = self.yListPixelsPerRegion[Cnt+1]
            idxRight += idxOffset
            Cnt += 1

        self.xBoundaryList = [self.bufferArray[self.ppAirBuffer - 1],
                              self.toothArray[self.ppEndTooth - 1]] + self.slotArray[self.ppSlot - 1::self.ppSlot] \
                             + self.toothArray[self.ppEndTooth + self.ppTooth - 1:-self.ppEndTooth:self.ppTooth] + [self.toothArray[-1],
                                                                                                                    self.bufferArray[-1]]

        a, b = 0, 0
        c, d = 0, 0
        Cnt, yCnt = 0, 0
        # Assign spatial data to the nodes
        while a < self.ppH:

            xCnt = 0
            # Keep track of the y coordinate for each node
            if a in self.yFirstEdgeNodes:
                delY = self.SpacingY / self.meshDensity[0]
            elif a in self.ySecondEdgeNodes:
                delY = self.SpacingY / self.meshDensity[1]
            else:
                delY = self.yMeshSizes[c]

            while b < self.ppL:

                # Keep track of the x coordinate for each node
                if b in self.xFirstEdgeNodes:
                    delX = self.SpacingX / self.meshDensity[0]
                elif b in self.xSecondEdgeNodes:
                    delX = self.SpacingX / self.meshDensity[1]
                else:
                    delX = self.xMeshSizes[d]

                self.matrix[a][b] = Node.buildFromScratch(iIndex=[b, a], iXinfo=[xCnt, delX], iYinfo=[yCnt, delY],
                                                          modelDepth=self.D)

                # Keep track of the x coordinate for each node
                if b in self.xFirstEdgeNodes:
                    xCnt += self.SpacingX / self.meshDensity[0]
                elif b in self.xSecondEdgeNodes:
                    xCnt += self.SpacingX / self.meshDensity[1]
                else:
                    xCnt += delX

                if b in self.xBoundaryList:
                    d += 1

                b += 1
                Cnt += 1
            d = 0

            # Keep track of the y coordinate for each node
            if a in self.yFirstEdgeNodes:
                yCnt += self.SpacingY / self.meshDensity[0]
            elif a in self.ySecondEdgeNodes:
                yCnt += self.SpacingY / self.meshDensity[1]
            else:
                yCnt += delY

            # TODO We cannot have the last row in this list so it must be unwritten
            if a in self.yBoundaryList:
                c += 1

            b = 0
            a += 1

        # Assign property data to the nodes
        a, b = 0, 0
        while a < self.ppH:
            while b < self.ppL:
                if a in self.yIndexesVacLower or a in self.yIndexesVacUpper:
                    self.matrix[a][b].material = 'vacuum'
                    self.matrix[a][b].ur = self.air.ur
                    self.matrix[a][b].sigma = self.air.sigma
                elif a in self.yIndexesYoke and b not in self.bufferArray:
                    self.matrix[a][b].material = 'iron'
                    self.matrix[a][b].ur = self.iron.ur
                    self.matrix[a][b].sigma = self.iron.sigma
                elif a in self.yIndexesAirGap:
                    self.matrix[a][b].material = 'vacuum'
                    self.matrix[a][b].ur = self.air.ur
                    self.matrix[a][b].sigma = self.air.sigma
                elif a in self.yIndexesBladeRotor:
                    self.matrix[a][b].material = 'aluminum'
                    self.matrix[a][b].ur = self.alum.ur
                    self.matrix[a][b].sigma = self.alum.sigma
                elif a in self.yIndexesBackIron:
                    self.matrix[a][b].material = 'iron'
                    self.matrix[a][b].ur = self.iron.ur
                    self.matrix[a][b].sigma = self.iron.sigma
                else:
                    if a in self.yIndexesUpperSlot:
                        aIdx = self.upper_slotsA
                        bIdx = self.upper_slotsB
                        cIdx = self.upper_slotsC
                    elif a in self.yIndexesLowerSlot:
                        aIdx = self.lower_slotsA
                        bIdx = self.lower_slotsB
                        cIdx = self.lower_slotsC
                    else:
                        aIdx = []
                        bIdx = []
                        cIdx = []

                    if b in self.toothArray:
                        self.matrix[a][b].material = 'iron'
                        self.matrix[a][b].ur = self.iron.ur
                        self.matrix[a][b].sigma = self.iron.sigma
                    elif b in self.bufferArray:
                        self.matrix[a][b].material = 'vacuum'
                        self.matrix[a][b].ur = self.air.ur
                        self.matrix[a][b].sigma = self.air.sigma
                    elif b in aIdx:
                        self.matrix[a][b].material = 'copperA'
                        self.matrix[a][b].ur = self.copper.ur
                        self.matrix[a][b].sigma = self.copper.sigma
                    elif b in bIdx:
                        self.matrix[a][b].material = 'copperB'
                        self.matrix[a][b].ur = self.copper.ur
                        self.matrix[a][b].sigma = self.copper.sigma
                    elif b in cIdx:
                        self.matrix[a][b].material = 'copperC'
                        self.matrix[a][b].ur = self.copper.ur
                        self.matrix[a][b].sigma = self.copper.sigma
                    elif b in self.removeLowerCoilIdxs + self.removeUpperCoilIdxs:
                        self.matrix[a][b].material = 'vacuum'
                        self.matrix[a][b].ur = self.air.ur
                        self.matrix[a][b].sigma = self.air.sigma
                    else:
                        self.matrix[a][b].material = ''
                b += 1
            b = 0
            a += 1

    def finalizeGrid(self):
        spatialDomainFlag = False

        # Define Indexes
        idxLeftAirBuffer = 0
        idxLeftEndTooth = idxLeftAirBuffer + self.ppAirBuffer
        idxSlot = idxLeftEndTooth + self.ppEndTooth
        idxTooth = idxSlot + self.ppSlot
        idxRightEndTooth = self.ppL - self.ppAirBuffer - self.ppEndTooth
        idxRightAirBuffer = idxRightEndTooth + self.ppEndTooth

        self.checkSpatialMapping(spatialDomainFlag, [idxLeftAirBuffer, idxLeftEndTooth, idxSlot, idxTooth, idxRightEndTooth, idxRightAirBuffer])

        # Scaling values for MMF-source distribution in section 2.2, equation 18, figure 5
        fraction = 0.5
        doubleBias = self.ppSlotHeight - fraction
        doubleCoilScaling = np.arange(fraction, doubleBias + fraction, 1)
        if self.invertY:
            doubleCoilScaling = np.flip(doubleCoilScaling)

        scalingLower, scalingUpper = 0.0, 0.0
        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        turnAreaRatio = self.ppSlot
        i, j = 0, 0
        tempTesting = 1
        while i < self.ppH:
            while j < self.ppL:

                self.matrix[i][j].Rx, self.matrix[i][j].Ry = self.matrix[i][j].getReluctance(self)

                isCurrentCu = self.matrix[i][j].material[:-1] == 'copper'
                if i in self.yIndexesLowerSlot and j in self.slotArray:
                    if j in self.lower_slotsA:
                        angle_plex = cmath.exp(0)
                    elif j in self.lower_slotsB:
                        angle_plex = cmath.exp(-j_plex * pi * 2 / 3)
                    elif j in self.lower_slotsC:
                        angle_plex = cmath.exp(j_plex * pi * 2 / 3)
                    else:
                        angle_plex = 0.0

                    # Set the scaling factor for MMF in equation 18
                    if isCurrentCu:
                        index_ = self.yIndexesLowerSlot.index(i)
                        if self.invertY:
                            index_ += len(doubleCoilScaling) // 2
                        scalingLower = doubleCoilScaling[index_]
                    else:
                        scalingLower = 0.0

                elif i in self.yIndexesUpperSlot and j in self.slotArray:
                    if j in self.upper_slotsA:
                        angle_plex = cmath.exp(0)
                    elif j in self.upper_slotsB:
                        angle_plex = cmath.exp(-j_plex * pi * 2 / 3)
                    elif j in self.upper_slotsC:
                        angle_plex = cmath.exp(j_plex * pi * 2 / 3)
                    else:
                        angle_plex = 0.0

                    # Set the scaling factor for MMF in equation 18
                    # 2 coils in slot
                    yLowerCoilIdx = i + self.ppSlotHeight // 2 if self.invertY else i - self.ppSlotHeight // 2
                    isLowerCu = self.matrix[yLowerCoilIdx][j].material[:-1] == 'copper'
                    if isCurrentCu and isLowerCu:
                        index_ = self.yIndexesUpperSlot.index(i)
                        if not self.invertY:
                            index_ += len(doubleCoilScaling) // 2
                        scalingUpper = doubleCoilScaling[index_]
                    # coil in upper slot only
                    elif isCurrentCu and not isLowerCu:
                        index_ = self.yIndexesUpperSlot.index(i)
                        if self.invertY:
                            index_ -= len(doubleCoilScaling) // 2
                        scalingUpper = doubleCoilScaling[index_]
                    else:
                        scalingUpper = 0.0

                else:
                    angle_plex = 0.0
                    scalingLower = 0.0
                    scalingUpper = 0.0

                self.matrix[i][j].Iph = self.Ip * angle_plex * time_plex

                # Lower slots only
                if i in self.yIndexesLowerSlot and j in self.slotArray:
                    if j in self.inLower_slotsA or j in self.inLower_slotsB or j in self.inLower_slotsC:
                        inOutCoeffMMF = -1
                    elif j in self.outLower_slotsA or j in self.outLower_slotsB or j in self.outLower_slotsC:
                        inOutCoeffMMF = 1
                    else:
                        if j not in self.removeLowerCoilIdxs:
                            print('Shouldnt happen')
                        inOutCoeffMMF = 0

                    self.matrix[i][j].MMF = inOutCoeffMMF * scalingLower * self.N * self.matrix[i][j].Iph / (2 * turnAreaRatio)

                # Upper slots only
                elif i in self.yIndexesUpperSlot and j in self.slotArray:
                    if j in self.inUpper_slotsA or j in self.inUpper_slotsB or j in self.inUpper_slotsC:
                        inOutCoeffMMF = -1
                    elif j in self.outUpper_slotsA or j in self.outUpper_slotsB or j in self.outUpper_slotsC:
                        inOutCoeffMMF = 1
                    else:
                        inOutCoeffMMF = 0

                    self.matrix[i][j].MMF = inOutCoeffMMF * scalingUpper * self.N * self.matrix[i][j].Iph / (2 * turnAreaRatio)

                # Stator teeth
                else:
                    self.matrix[i][j].MMF = 0.0

                # # TODO - This is temp
                # if j in self.outLower_slotsC[1]:
                #     self.matrix[i][j].MMF *= 1

                j += 1
            j = 0
            i += 1

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
    def checkSpatialMapping(self, spatialDomainFlag, iIdxs):

        iIdxLeftAirBuffer, iIdxLeftEndTooth, iIdxSlot, iIdxTooth, iIdxRightEndTooth, iIdxRightAirBuffer = iIdxs

        # The difference between expected and actual is rounded due to quantization error
        #  in the 64 bit floating point arithmetic

        # X direction Checks

        yIdx = 0
        # Check slotpitch
        if round(self.slotpitch - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxSlot].x), 12) != 0:
            print(f'flag - slotpitch: {self.slotpitch - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxSlot].x)}')
            spatialDomainFlag = True
        # Check slot width
        if round(self.ws - (self.matrix[yIdx][iIdxTooth].x - self.matrix[yIdx][iIdxSlot].x), 12) != 0:
            print(f'flag - slots: {self.ws - (self.matrix[yIdx][iIdxTooth].x - self.matrix[yIdx][iIdxSlot].x)}')
            spatialDomainFlag = True
        # Check tooth width
        if round(self.wt - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxTooth].x), 12) != 0:
            print(f'flag - teeth: {self.wt - (self.matrix[yIdx][iIdxTooth + self.ppTooth].x - self.matrix[yIdx][iIdxTooth].x)}')
            spatialDomainFlag = True
        # Check left end tooth
        if round(self.endTooth - (self.matrix[yIdx][iIdxSlot].x - self.matrix[yIdx][iIdxLeftEndTooth].x), 12) != 0:
            print(f'flag - left end tooth: {self.endTooth - (self.matrix[yIdx][iIdxSlot].x - self.matrix[yIdx][iIdxLeftEndTooth].x)}')
            spatialDomainFlag = True
        # Check right end tooth
        if round(self.endTooth - (self.matrix[yIdx][iIdxRightAirBuffer].x - self.matrix[yIdx][iIdxRightEndTooth].x), 12) != 0:
            print(f'flag - right end tooth: {self.endTooth - (self.matrix[yIdx][iIdxRightAirBuffer].x - self.matrix[yIdx][iIdxRightEndTooth].x)}')
            spatialDomainFlag = True
        # Check left air buffer
        if round(self.Airbuffer - (self.matrix[yIdx][iIdxLeftEndTooth].x - self.matrix[yIdx][iIdxLeftAirBuffer].x), 12) != 0:
            print(f'flag - left air buffer: {self.Airbuffer - (self.matrix[yIdx][iIdxLeftEndTooth].x - self.matrix[yIdx][iIdxLeftAirBuffer].x)}')
            spatialDomainFlag = True
        # Check right air buffer
        if round(self.Airbuffer - (self.matrix[yIdx][-1].x + self.matrix[yIdx][-1].lx - self.matrix[yIdx][iIdxRightAirBuffer].x), 12) != 0:
            print(f'flag - right air buffer: {self.Airbuffer - (self.matrix[yIdx][-1].x + self.matrix[yIdx][-1].lx - self.matrix[yIdx][iIdxRightAirBuffer].x)}')
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


class Node(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for attr_key in kwargs:
                if type(kwargs[attr_key]) == list and kwargs[attr_key][0] == 'plex_Signature':
                    self.__dict__[attr_key] = rebuildPlex(kwargs[attr_key])
                else:
                    self.__dict__[attr_key] = kwargs[attr_key]
            return

        self.xIndex = kwargs['iIndex'][0]
        self.yIndex = kwargs['iIndex'][1]

        # Initialize dense meshing near slot and teeth edges in x direction
        self.x = kwargs['iXinfo'][0]
        self.lx = kwargs['iXinfo'][1]

        # Initialize dense meshing near slot and teeth edges in y direction
        self.y = kwargs['iYinfo'][0]
        self.ly = kwargs['iYinfo'][1]

        # Cross sectional area
        self.Szy = self.ly*kwargs['modelDepth']
        self.Sxz = self.lx*kwargs['modelDepth']

        self.xCenter = self.x + self.lx / 2
        self.yCenter = self.y + self.ly / 2

        # Node properties
        self.ur = 0.0
        self.sigma = 0.0
        self.material = ''
        self.colour = ''

        # Potential
        self.Yk = np.cdouble(0.0)

        # B field
        self.Bx, self.By, self.B = np.zeros(3, dtype=np.cdouble)

        # Flux
        self.phiXp, self.phiXn, self.phiYp, self.phiYn, self.phiError = np.zeros(5, dtype=np.cdouble)

        # Current
        self.Iph = np.cdouble(0.0)  # phase

        # MMF
        self.MMF = np.cdouble(0.0)  # AmpereTurns

        # Reluctance
        self.Rx, self.Ry = np.zeros(2, dtype=np.float64)

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


def meshBoundary(spatial, pixels, spacing, fraction, numBoundaries, meshDensity):

    meshSize = (spatial - numBoundaries*fraction*spacing) / (pixels - len(meshDensity)*numBoundaries)

    return meshSize
