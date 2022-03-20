from builtins import list
from numpy.core._multiarray_umath import ndarray
from LIM.SlotPoleCalculation import*
from collections import deque


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

    def __init__(self, kwargs):

        super().__init__(kwargs['slots'], kwargs['poles'], kwargs['length'])

        xMeshIndexes = kwargs['meshIndexes'][0]
        yMeshIndexes = kwargs['meshIndexes'][1]

        self.Spacing = kwargs['pixelSpacing']
        self.Cspacing = kwargs['canvasSpacing'] / self.H
        self.meshDensity = kwargs['meshDensity']
        self.n = kwargs['n']
        self.typeList = []
        self.complexTypeList = []

        # X-direction
        self.ppSlotpitch = self.setPixelsPerLength(length=self.slotpitch, minimum=2)
        self.ppAirBuffer = self.setPixelsPerLength(length=self.Airbuffer, minimum=1)
        self.ppTooth = self.setPixelsPerLength(length=self.wt, minimum=1)
        self.ppSlot = self.ppSlotpitch - self.ppTooth
        self.ppLeftEndTooth = self.setPixelsPerLength(length=self.endTeeth, minimum=1)
        self.ppRightEndTooth = self.ppLeftEndTooth
        # Y-direction
        self.ppVacuumLower = self.setPixelsPerLength(length=self.vac, minimum=1)
        self.ppVacuumUpper = self.ppVacuumLower
        self.ppYokeheight = self.setPixelsPerLength(length=self.hy, minimum=1)
        self.ppAirgap = self.setPixelsPerLength(length=self.g, minimum=1)
        self.ppBladerotor = self.setPixelsPerLength(length=self.dr, minimum=1)
        self.ppBackIron = self.setPixelsPerLength(length=self.bi, minimum=1)
        self.ppSlotheight = self.setPixelsPerLength(length=self.hs, minimum=2)

        if self.ppSlotheight % 2 != 0:
            self.ppSlotheight += 1

        self.ppHeight = self.ppYokeheight + self.ppSlotheight
        self.ppLength = (self.slots - 1) * self.ppSlotpitch + self.ppSlot + self.ppLeftEndTooth + self.ppRightEndTooth + 2 * self.ppAirBuffer
        self.matrix = np.array([[type('', (Node,), {}) for _ in range(self.ppLength)] for _ in range(self.ppHeight + self.ppAirgap + self.ppBladerotor + self.ppBackIron + self.ppVacuumLower + self.ppVacuumUpper)])
        self.ppL = len(self.matrix[0])
        self.ppH = len(self.matrix)

        self.toothArray, self.coilArray, self.bufferArray = np.zeros((3, 1), dtype=np.int16)
        self.removeLowerCoils, self.removeUpperCoils = np.zeros((2, 1), dtype=np.int16)
        self.removeLowerCoilIdxs, self.removeUpperCoilIdxs = [], []
        self.xFirstEdgeNodes, self.xSecondEdgeNodes = [], []
        self.yFirstEdgeNodes, self.ySecondEdgeNodes = [], []

        # These lists represent whether a region boundary will have dense meshing or not.
        #  Where each region is in groups of two [#, #], one for each boundary meshing (1=use mesh density, 0=don't)
        self.xMeshSizes = np.zeros(len(xMeshIndexes), dtype=np.float64)
        self.yMeshSizes = np.zeros(len(yMeshIndexes), dtype=np.float64)

        # Mesh sizing
        self.fractionSize = (1 / self.meshDensity[0] + 1 / self.meshDensity[1])
        self.xListSpatialDatum = [self.Airbuffer, self.endTeeth] + [self.ws, self.wt] * (self.slots - 1) + [self.ws, self.endTeeth, self.Airbuffer]
        self.xListPixelsPerRegion = [self.ppAirBuffer, self.ppLeftEndTooth] + [self.ppSlot, self.ppTooth] * (self.slots - 1) + [self.ppSlot, self.ppRightEndTooth, self.ppAirBuffer]
        # self.yListSpatialDatum = [self.vac, self.hy, self.hs/2, self.hs/2, self.g, self.dr, self.bi, self.vac]
        # self.yListPixelsPerRegion = [self.ppVacuumLower, self.ppYokeheight, self.ppSlotheight//2, self.ppSlotheight//2, self.ppAirgap, self.ppBladerotor, self.ppBackIron, self.ppVacuumUpper]
        self.yListSpatialDatum = [self.vac, self.bi, self.dr, self.g, self.hs / 2, self.hs / 2, self.hy, self.vac]
        self.yListPixelsPerRegion = [self.ppVacuumLower, self.ppBackIron, self.ppBladerotor, self.ppAirgap, self.ppSlotheight//2, self.ppSlotheight//2, self.ppYokeheight, self.ppVacuumUpper]

        for Cnt in range(len(self.xMeshSizes)):
            self.xMeshSizes[Cnt] = meshBoundary(self.xListSpatialDatum[Cnt], self.xListPixelsPerRegion[Cnt], self.Spacing, self.fractionSize, sum(xMeshIndexes[Cnt]), self.meshDensity)
            if self.xMeshSizes[Cnt] < 0:
                print('negative x mesh sizes', Cnt)
                return

        for Cnt in range(len(self.yMeshSizes)):
            self.yMeshSizes[Cnt] = meshBoundary(self.yListSpatialDatum[Cnt], self.yListPixelsPerRegion[Cnt], self.Spacing, self.fractionSize, sum(yMeshIndexes[Cnt]), self.meshDensity)
            if self.yMeshSizes[Cnt] < 0:
                print('negative y mesh sizes')
                return

        self.hmRegions = kwargs['hmRegions']
        self.mecRegions = kwargs['mecRegions']
        self.hmRegionsIndex = np.zeros(len(self.hmRegions) + 1, dtype=np.int32)
        self.mecRegionsIndex = np.zeros(len(self.mecRegions), dtype=np.int32)

        self.mecRegionLength = self.matrix[self.ppVacuumLower:self.ppHeight + self.ppVacuumLower, :].size

        # Update the hm and mec RegionIndex
        self.setRegionIndices()

        ironOffset = self.ppVacuumLower
        bladerRotorOffset = ironOffset + self.ppBackIron
        airgapOffset = bladerRotorOffset + self.ppBladerotor
        slotOffset = airgapOffset + self.ppAirgap
        yokeOffset = slotOffset + self.ppSlotheight

        self.yIndexesVacLower = list(range(0, self.ppVacuumLower))
        self.yIndexesBackIron = list(range(ironOffset, ironOffset + self.ppBackIron))
        self.yIndexesBladeRotor = list(range(bladerRotorOffset, bladerRotorOffset + self.ppBladerotor))
        self.yIndexesAirgap = list(range(airgapOffset, airgapOffset + self.ppAirgap))
        self.yIndexesLowerSlot = list(range(slotOffset, slotOffset + self.ppSlotheight // 2))
        self.yIndexesUpperSlot = list(range(slotOffset + self.ppSlotheight // 2, slotOffset + self.ppSlotheight))
        self.yIndexesYoke = list(range(yokeOffset, yokeOffset + self.ppYokeheight))
        self.yIndexesVacUpper = list(range(self.ppH - self.ppVacuumUpper, self.ppH))
        self.yIndexesHM = self.yIndexesVacLower + self.yIndexesBackIron + self.yIndexesBladeRotor + self.yIndexesAirgap + self.yIndexesVacUpper
        self.yIndexesMEC = self.yIndexesLowerSlot + self.yIndexesUpperSlot + self.yIndexesYoke

        # Thrust of the entire integration region
        self.Fx = 0.0
        self.Fy = 0.0

    def buildGrid(self, pixelSpacing, meshIndexes):
        xMeshIndexes = meshIndexes[0]
        yMeshIndexes = meshIndexes[1]

        #  Initialize grid with Nodes
        a, b = 0, 0
        slotCount = 0
        listOffset = self.ppAirBuffer + self.ppLeftEndTooth
        oldTooth, newTooth, oldSlot, newSlot = 0, listOffset, 0, 0
        fullToothArray, slotArray = [], []
        while a < self.ppH:
            while b < self.ppL:
                # Create indexes for slots and teeth
                if slotCount < self.slots and listOffset < b < self.ppL-listOffset:
                    oldSlot = newTooth
                    newSlot = oldSlot + self.ppSlot
                    oldTooth = newSlot
                    newTooth = oldTooth + self.ppTooth
                    if slotCount < self.slots - 1:
                        fullToothArray += list(range(oldTooth, newTooth))
                    slotCount += 1
                else:
                    pass
                b += 1
            b = 0
            a += 1

        leftEndTooth = list(range(self.ppAirBuffer, listOffset))
        rightEndTooth = list(range(fullToothArray[-1] + self.ppSlot + 1, fullToothArray[-1] + self.ppSlot + self.ppRightEndTooth + 1))
        self.toothArray = leftEndTooth + fullToothArray + rightEndTooth
        self.coilArray = [x for x in range(self.ppAirBuffer, self.ppL - self.ppAirBuffer) if x not in self.toothArray]
        self.bufferArray = [x for x in list(range(self.ppL)) if x not in self.coilArray and x not in self.toothArray]

        offset = self.ppSlot*math.ceil(self.q)
        intSlots = math.ceil(self.q)*self.poles*3

        # Split slots into respective phases
        windingShift = 2
        temp_upper_slotArray = self.coilArray
        upper_slotArray = temp_upper_slotArray + [None] * self.ppSlot * (intSlots - self.slots)
        lower_slotArray = deque(upper_slotArray)
        lower_slotArray.rotate(-windingShift*offset)
        lower_slotArray = list(lower_slotArray)

        upper_slotArrayA, upper_slotArrayB, upper_slotArrayC = [], [], []
        lower_slotArrayA, lower_slotArrayB, lower_slotArrayC = [], [], []
        for threeSlots in range(0, self.slots, 3):
            upper_slotArrayA += upper_slotArray[(threeSlots+1)*offset:(threeSlots+2)*offset]
            upper_slotArrayB += upper_slotArray[threeSlots*offset:(threeSlots+1)*offset]
            upper_slotArrayC += upper_slotArray[(threeSlots+2)*offset:(threeSlots+3)*offset]

            lower_slotArrayA += lower_slotArray[(threeSlots+1)*offset:(threeSlots+2)*offset]
            lower_slotArrayB += lower_slotArray[threeSlots*offset:(threeSlots+1)*offset]
            lower_slotArrayC += lower_slotArray[(threeSlots+2)*offset:(threeSlots+3)*offset]

        self.removeLowerCoils = [0, 1, 2, self.slots-1]
        self.removeUpperCoils = [0, self.slots-1, self.slots-2, self.slots-3]

        for idx in self.removeLowerCoils:
            coilOffset = idx*self.ppSlot
            self.removeLowerCoilIdxs += self.coilArray[coilOffset:coilOffset+self.ppSlot]

        for idx in self.removeUpperCoils:
            coilOffset = idx*self.ppSlot
            self.removeUpperCoilIdxs += self.coilArray[coilOffset:coilOffset+self.ppSlot]

        upper_slotArrayA = [x for x in upper_slotArrayA if x not in self.removeUpperCoilIdxs]
        upper_slotArrayB = [x for x in upper_slotArrayB if x not in self.removeUpperCoilIdxs]
        upper_slotArrayC = [x for x in upper_slotArrayC if x not in self.removeUpperCoilIdxs]
        lower_slotArrayA = [x for x in lower_slotArrayA if x not in self.removeLowerCoilIdxs]
        lower_slotArrayB = [x for x in lower_slotArrayB if x not in self.removeLowerCoilIdxs]
        lower_slotArrayC = [x for x in lower_slotArrayC if x not in self.removeLowerCoilIdxs]

        self.upper_slotsA = np.array(upper_slotArrayA)
        self.lower_slotsA = np.array(lower_slotArrayA)

        self.upper_slotsB = np.array(upper_slotArrayB)
        self.lower_slotsB = np.array(lower_slotArrayB)

        self.lower_slotsC = np.array(lower_slotArrayC)
        self.upper_slotsC = np.array(upper_slotArrayC)

        # Remove None from
        lowerCoilsA_NoneRemoved = list(filter(None, self.lower_slotsA))
        lowerCoilsA = np.split(np.array(lowerCoilsA_NoneRemoved), len(lowerCoilsA_NoneRemoved)//self.ppSlot, axis=0)
        upperCoilsA_NoneRemoved = list(filter(None, self.upper_slotsA))
        upperCoilsA = np.split(np.array(upperCoilsA_NoneRemoved), len(upperCoilsA_NoneRemoved)//self.ppSlot, axis=0)

        lowerCoilsB_NoneRemoved = list(filter(None, self.lower_slotsB))
        lowerCoilsB = np.split(np.array(lowerCoilsB_NoneRemoved), len(lowerCoilsB_NoneRemoved)//self.ppSlot, axis=0)
        upperCoilsB_NoneRemoved = list(filter(None, self.upper_slotsB))
        upperCoilsB = np.split(np.array(upperCoilsB_NoneRemoved), len(upperCoilsB_NoneRemoved)//self.ppSlot, axis=0)

        lowerCoilsC_NoneRemoved = list(filter(None, self.lower_slotsC))
        lowerCoilsC = np.split(np.array(lowerCoilsC_NoneRemoved), len(lowerCoilsC_NoneRemoved)//self.ppSlot, axis=0)
        upperCoilsC_NoneRemoved = list(filter(None, self.upper_slotsC))
        upperCoilsC = np.split(np.array(upperCoilsC_NoneRemoved), len(upperCoilsC_NoneRemoved)//self.ppSlot, axis=0)

        # Sort coils into direction of current (ex, in and out of page)
        self.inLower_slotsA = np.array(lowerCoilsA[1::2])
        self.outLower_slotsA = np.array(lowerCoilsA[::2])
        self.inUpper_slotsA = np.array(upperCoilsA[1::2])
        self.outUpper_slotsA = np.array(upperCoilsA[::2])
        self.inLower_slotsB = np.array(lowerCoilsB[1::2])
        self.outLower_slotsB = np.array(lowerCoilsB[::2])
        self.inUpper_slotsB = np.array(upperCoilsB[1::2])
        self.outUpper_slotsB = np.array(upperCoilsB[::2])
        self.inLower_slotsC = np.array(lowerCoilsC[::2])
        self.outLower_slotsC = np.array(lowerCoilsC[1::2])
        self.inUpper_slotsC = np.array(upperCoilsC[::2])
        self.outUpper_slotsC = np.array(upperCoilsC[1::2])

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

        xBoundaryList = [self.bufferArray[self.ppAirBuffer - 1], self.toothArray[self.ppLeftEndTooth - 1]] + self.coilArray[self.ppSlot-1::self.ppSlot] + self.toothArray[self.ppLeftEndTooth + self.ppTooth - 1:-self.ppRightEndTooth:self.ppTooth] + [self.toothArray[-1], self.bufferArray[-1]]
        yVac1Boundary = self.ppVacuumLower-1
        yBackIronBoundary = yVac1Boundary + self.ppBackIron
        yBladeBoundary = yBackIronBoundary + self.ppBladerotor
        yAirBoundary = yBladeBoundary + self.ppAirgap
        yLowerCoilsBoundary = yAirBoundary + self.ppSlotheight//2
        yUpperCoilsBoundary = yLowerCoilsBoundary + self.ppSlotheight//2
        yYokeBoundary = yUpperCoilsBoundary + self.ppYokeheight
        yVac2Boundary = yYokeBoundary + self.ppVacuumUpper

        yBoundaryList = [yVac1Boundary, yBackIronBoundary, yBladeBoundary, yAirBoundary, yLowerCoilsBoundary, yUpperCoilsBoundary, yYokeBoundary, yVac2Boundary]

        a, b = 0, 0
        c, d, e, f = 0, 0, 0, 0
        Cnt = 0
        yCnt = 0

        # Assign spatial data to the nodes
        while a < self.ppH:

            xCnt = 0
            # Keep track of the y coordinate for each node
            if a in self.yFirstEdgeNodes:
                delY = pixelSpacing / self.meshDensity[0]
            elif a in self.ySecondEdgeNodes:
                delY = pixelSpacing / self.meshDensity[1]
            else:
                # Vacuum lower
                if a in self.yIndexesVacLower:
                    delY = self.yMeshSizes[d]
                # Yoke
                elif a in self.yIndexesYoke:
                    delY = self.yMeshSizes[d]
                # Lower coils
                elif a in self.yIndexesLowerSlot:
                    delY = self.yMeshSizes[d]
                # Upper coils
                elif a in self.yIndexesUpperSlot:
                    delY = self.yMeshSizes[d]
                # Air gap
                elif a in self.yIndexesAirgap:
                    delY = self.yMeshSizes[d]
                # Blade rotor
                elif a in self.yIndexesBladeRotor:
                    delY = self.yMeshSizes[d]
                # Back iron
                elif a in self.yIndexesBackIron:
                    delY = self.yMeshSizes[d]
                # Vacuum upper
                elif a in self.yIndexesVacUpper:
                    delY = self.yMeshSizes[d]
                else:
                    delY = pixelSpacing

            while b < self.ppL:

                # Keep track of the x coordinate for each node
                if b in self.xFirstEdgeNodes:
                    delX = pixelSpacing / self.meshDensity[0]
                elif b in self.xSecondEdgeNodes:
                    delX = pixelSpacing / self.meshDensity[1]
                else:
                    # Air buffer
                    if b in self.bufferArray:
                        delX = self.xMeshSizes[c]
                    # End teeth
                    elif b in self.toothArray[:self.ppLeftEndTooth] + self.toothArray[-self.ppRightEndTooth:]:
                        delX = self.xMeshSizes[c]
                    # Coils
                    elif b in self.coilArray:
                        delX = self.xMeshSizes[c]
                    # Full teeth
                    elif b in self.toothArray and b not in self.toothArray[:self.ppLeftEndTooth] + self.toothArray[-self.ppRightEndTooth:]:
                        delX = self.xMeshSizes[c]
                    else:
                        delX = pixelSpacing

                # delX can be negative which means x can be 0
                self.matrix[a][b] = Node.buildFromScratch(iIndex=[b, a], iXinfo=[xCnt, delX], iYinfo=[yCnt, delY], modelDepth=self.D)

                # Keep track of the x coordinate for each node
                if b in self.xFirstEdgeNodes:
                    xCnt += pixelSpacing / self.meshDensity[0]
                elif b in self.xSecondEdgeNodes:
                    xCnt += pixelSpacing / self.meshDensity[1]
                else:
                    # Air buffer
                    if b in self.bufferArray:
                        xCnt += self.xMeshSizes[e]
                    # End teeth
                    elif b in self.toothArray[:self.ppLeftEndTooth] + self.toothArray[-self.ppRightEndTooth:]:
                        xCnt += self.xMeshSizes[e]
                    # Coils
                    elif b in self.coilArray:
                        xCnt += self.xMeshSizes[e]
                    # Full teeth
                    elif b in self.toothArray and b not in self.toothArray[:self.ppLeftEndTooth] + self.toothArray[-self.ppRightEndTooth:]:
                        xCnt += self.xMeshSizes[e]
                    else:
                        xCnt += pixelSpacing

                if b in xBoundaryList:
                    c += 1
                    e += 1

                b += 1
                Cnt += 1
            c, e = 0, 0

            # Keep track of the y coordinate for each node
            if a in self.yFirstEdgeNodes:
                yCnt += pixelSpacing / self.meshDensity[0]
            elif a in self.ySecondEdgeNodes:
                yCnt += pixelSpacing / self.meshDensity[1]
            else:
                # Vacuum lower
                if a in self.yIndexesVacLower:
                    yCnt += self.yMeshSizes[f]
                # Yoke
                elif a in self.yIndexesYoke:
                    yCnt += self.yMeshSizes[f]
                # Lower coils
                elif a in self.yIndexesLowerSlot:
                    yCnt += self.yMeshSizes[f]
                # Upper coils
                elif a in self.yIndexesUpperSlot:
                    yCnt += self.yMeshSizes[f]
                # Air gap
                elif a in self.yIndexesAirgap:
                    yCnt += self.yMeshSizes[f]
                # Blade rotor
                elif a in self.yIndexesBladeRotor:
                    yCnt += self.yMeshSizes[f]
                # Back iron
                elif a in self.yIndexesBackIron:
                    yCnt += self.yMeshSizes[f]
                # Vacuum upper
                elif a in self.yIndexesVacUpper:
                    yCnt += self.yMeshSizes[f]
                else:
                    yCnt += pixelSpacing

            if a in yBoundaryList:
                d += 1
                f += 1

            b = 0
            a += 1

        # Assign property data to the nodes
        a, b = 0, 0
        while a < self.ppH:
            while b < self.ppL:
                if a in self.yIndexesYoke and b not in self.bufferArray:
                    self.matrix[a][b].material = 'iron'
                    self.matrix[a][b].ur = self.ur_iron
                    self.matrix[a][b].sigma = self.sigma_iron
                elif a in self.yIndexesAirgap:
                    self.matrix[a][b].material = 'vacuum'
                    self.matrix[a][b].ur = self.ur_air
                    self.matrix[a][b].sigma = self.sigma_air
                elif a in self.yIndexesVacLower or a in self.yIndexesVacUpper:
                    self.matrix[a][b].material = 'vacuum'
                    self.matrix[a][b].ur = self.ur_air
                    self.matrix[a][b].sigma = self.sigma_air
                elif a in self.yIndexesBladeRotor:
                    self.matrix[a][b].material = 'aluminum'
                    self.matrix[a][b].ur = self.ur_alum
                    self.matrix[a][b].sigma = self.sigma_alum
                elif a in self.yIndexesBackIron:
                    self.matrix[a][b].material = 'iron'
                    self.matrix[a][b].ur = self.ur_iron
                    self.matrix[a][b].sigma = self.sigma_iron
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
                        self.matrix[a][b].ur = self.ur_iron
                        self.matrix[a][b].sigma = self.sigma_iron
                    elif b in self.bufferArray:
                        self.matrix[a][b].material = 'vacuum'
                        self.matrix[a][b].ur = self.ur_air
                        self.matrix[a][b].sigma = self.sigma_air
                    elif b in aIdx:
                        self.matrix[a][b].material = 'copperA'
                        self.matrix[a][b].ur = self.ur_copp
                        self.matrix[a][b].sigma = self.sigma_copp
                    elif b in bIdx:
                        self.matrix[a][b].material = 'copperB'
                        self.matrix[a][b].ur = self.ur_copp
                        self.matrix[a][b].sigma = self.sigma_copp
                    elif b in cIdx:
                        self.matrix[a][b].material = 'copperC'
                        self.matrix[a][b].ur = self.ur_copp
                        self.matrix[a][b].sigma = self.sigma_copp
                    elif b in self.removeLowerCoilIdxs + self.removeUpperCoilIdxs:
                        self.matrix[a][b].material = 'vacuum'
                        self.matrix[a][b].ur = self.ur_air
                        self.matrix[a][b].sigma = self.sigma_air
                    else:
                        self.matrix[a][b].material = ''
                b += 1
            b = 0
            a += 1

    def finalizeGrid(self, pixelDivisions):
        spatialDomainFlag = False

        # Define Indexes
        idxLeftAirBuffer = 0
        idxLeftEndTooth = idxLeftAirBuffer + self.ppAirBuffer
        idxSlot = idxLeftEndTooth + self.ppLeftEndTooth
        idxTooth = idxSlot + self.ppSlot
        idxRightEndTooth = self.ppL - self.ppAirBuffer - self.ppRightEndTooth
        idxRightAirBuffer = idxRightEndTooth + self.ppRightEndTooth

        self.checkSpatialMapping(pixelDivisions, spatialDomainFlag,
                                 [idxLeftAirBuffer, idxLeftEndTooth, idxSlot, idxTooth, idxRightEndTooth,
                                  idxRightAirBuffer])

        # Scaling values for MMF-source distribution in section 2.2, equation 18, figure 5
        fraction = 0.5
        doubleBias = self.ppSlotheight - fraction
        doubleCoilScaling = [doubleBias - i if i > 0 else doubleBias for i in range(self.ppSlotheight)]

        scalingLower, scalingUpper = 0.0, 0.0
        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        turnAreaRatio = self.ppSlot
        i, j = 0, 0
        while i < self.ppH:
            while j < self.ppL:

                self.matrix[i][j].Rx, self.matrix[i][j].Ry = self.matrix[i][j].getReluctance()

                if i in self.yIndexesLowerSlot and j in self.coilArray:
                    if j in self.lower_slotsA:
                        angle_plex = cmath.exp(0)
                    elif j in self.lower_slotsB:
                        angle_plex = cmath.exp(-j_plex * pi * 2 / 3)
                    elif j in self.lower_slotsC:
                        angle_plex = cmath.exp(j_plex * pi * 2 / 3)
                    else:
                        angle_plex = 0.0

                    # Set the scaling factor for MMF in equation 18
                    # 2 coils in slot
                    if self.matrix[i][j].material[:-1] == 'copper' and self.matrix[i - self.ppSlotheight // 2][j].material[:-1] == 'copper':
                        index_ = self.yIndexesLowerSlot.index(i)
                        scalingLower = doubleCoilScaling[len(doubleCoilScaling) // 2 + index_]
                    # coil in upper slot only
                    elif self.matrix[i][j].material[:-1] != 'copper' and self.matrix[i - self.ppSlotheight // 2][j].material[:-1] == 'copper':
                        scalingLower = 0.0
                    # coil in lower slot only
                    elif self.matrix[i][j].material[:-1] == 'copper' and self.matrix[i - self.ppSlotheight // 2][j].material[:-1] != 'copper':
                        index_ = self.yIndexesLowerSlot.index(i)
                        scalingLower = doubleCoilScaling[len(doubleCoilScaling) // 2 + index_]
                    # empty slot
                    elif self.matrix[i][j].material == 'vacuum' and self.matrix[i - self.ppSlotheight // 2][j].material == 'vacuum':
                        scalingLower = 0.0
                    else:
                        scalingLower = 0.0

                elif i in self.yIndexesUpperSlot and j in self.coilArray:
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
                    if self.matrix[i][j].material[:-1] == 'copper' and self.matrix[i + self.ppSlotheight // 2][j].material[:-1] == 'copper':
                        index_ = self.yIndexesUpperSlot.index(i)
                        scalingUpper = doubleCoilScaling[index_]
                    # coil in lower slot only
                    elif self.matrix[i][j].material[:-1] != 'copper' and self.matrix[i + self.ppSlotheight // 2][j].material[:-1] == 'copper':
                        scalingUpper = 0
                    # coil in upper slot only
                    elif self.matrix[i][j].material[:-1] == 'copper' and self.matrix[i + self.ppSlotheight // 2][j].material[:-1] != 'copper':
                        index_ = self.yIndexesUpperSlot.index(i)
                        scalingUpper = doubleCoilScaling[len(doubleCoilScaling) // 2 + index_]
                    # empty slot
                    elif self.matrix[i][j].material == 'vacuum' and self.matrix[i + self.ppSlotheight // 2][j].material == 'vacuum':
                        scalingUpper = 0.0
                    else:
                        scalingUpper = 0.0

                else:
                    angle_plex = 0.0
                    scalingLower = 0.0
                    scalingUpper = 0.0

                self.matrix[i][j].Iph = self.Ip * angle_plex * time_plex

                # Lower slots only
                if i in self.yIndexesLowerSlot and j in self.coilArray:
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
                elif i in self.yIndexesUpperSlot and j in self.coilArray:
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

                j += 1
            j = 0
            i += 1

    def setPixelsPerLength(self, length, minimum):

        pixels = round(length / self.Spacing)
        return minimum if pixels < minimum else pixels

    def setRegionIndices(self):

        # Create the starting index for each region in the columns of matrix A
        regionIndex, hmCount, mecCount = 0, 0, 0
        for count in range(1, len(self.mecRegions) + len(self.hmRegions) + 2):
            if count in self.hmRegions:
                # Dirichlet boundaries which have half the unknown coefficients
                if self.hmRegions[count] == 'vac':
                    self.hmRegionsIndex[hmCount] = regionIndex
                    regionIndex += 2 * len(self.n)

                else:
                    self.hmRegionsIndex[hmCount] = regionIndex
                    regionIndex += 2 * len(self.n)
                hmCount += 1

            elif count in self.mecRegions:
                self.mecRegionsIndex[mecCount] = regionIndex
                regionIndex += self.mecRegionLength
                mecCount += 1

            elif count == list(self.hmRegions)[-1] + 1:
                self.hmRegionsIndex[hmCount] = regionIndex

            else:
                self.writeErrorToDict(key='name',
                                      error=Error.buildFromScratch(name='gridRegion',
                                                                   description='ERROR - Error in the iGrid regions list',
                                                                   cause=True))

    # This function is written to catch any errors in the mapping between canvas and space for both x and y coordinates
    def checkSpatialMapping(self, pixelDivision, spatialDomainFlag, iIdxs):

        iIdxLeftAirBuffer, iIdxLeftEndTooth, iIdxSlot, iIdxTooth, iIdxRightEndTooth, iIdxRightAirBuffer = iIdxs

        # The difference between expected and actual is rounded due to quantization error
        #  in the 64 bit floating point arithmetic

        # X direction Checks

        yIdx = 0
        # Check grid spacing to slotpitch
        if round(self.slotpitch - self.Spacing * pixelDivision, 12) != 0:
            print(f'flag - iGrid spacing: {self.slotpitch - self.Spacing * pixelDivision}')
            spatialDomainFlag = True
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
        if round(self.endTeeth - (self.matrix[yIdx][iIdxSlot].x - self.matrix[yIdx][iIdxLeftEndTooth].x), 12) != 0:
            print(f'flag - left end tooth: {self.endTeeth - (self.matrix[yIdx][iIdxSlot].x - self.matrix[yIdx][iIdxLeftEndTooth].x)}')
            spatialDomainFlag = True
        # Check right end tooth
        if round(self.endTeeth - (self.matrix[yIdx][iIdxRightAirBuffer].x - self.matrix[yIdx][iIdxRightEndTooth].x), 12) != 0:
            print(f'flag - right end tooth: {self.endTeeth - (self.matrix[yIdx][iIdxRightAirBuffer].x - self.matrix[yIdx][iIdxRightEndTooth].x)}')
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

        xIdx = 0
        # Check Vacuum Lower
        diffVacLowerDims = self.vac - (self.matrix[self.yIndexesBackIron[0]][xIdx].y - self.matrix[self.yIndexesVacLower[0]][xIdx].y)
        if round(diffVacLowerDims, 12) != 0:
            print(f'flag - vacuum lower: {diffVacLowerDims}')
            spatialDomainFlag = True
        # Check Back Iron
        diffBackIronDims = self.bi - (self.matrix[self.yIndexesBladeRotor[0]][xIdx].y - self.matrix[self.yIndexesBackIron[0]][xIdx].y)
        if round(diffBackIronDims, 12) != 0:
            print(f'flag - blade rotor: {diffBackIronDims}')
            spatialDomainFlag = True
        # Check Blade Rotor
        diffBladeRotorDims = self.dr - (self.matrix[self.yIndexesAirgap[0]][xIdx].y - self.matrix[self.yIndexesBladeRotor[0]][xIdx].y)
        if round(diffBladeRotorDims, 12) != 0:
            print(f'flag - blade rotor: {diffBladeRotorDims}')
            spatialDomainFlag = True
        # Check Air Gap
        diffAirGapDims = self.g - (self.matrix[self.yIndexesLowerSlot[0]][xIdx].y - self.matrix[self.yIndexesAirgap[0]][xIdx].y)
        if round(diffAirGapDims, 12) != 0:
            print(f'flag - air gap: {diffAirGapDims}')
            spatialDomainFlag = True
        # Check Slot/Tooth Height
        diffSlotHeightDims = self.hs - (self.matrix[self.yIndexesYoke[0]][xIdx].y - self.matrix[self.yIndexesLowerSlot[0]][xIdx].y)
        if round(diffSlotHeightDims, 12) != 0:
            print(f'flag - slot height: {diffSlotHeightDims}')
            spatialDomainFlag = True
        # Check Yoke
        diffYokeDims = self.hy - (self.matrix[self.yIndexesVacUpper[0]][xIdx].y - self.matrix[self.yIndexesYoke[0]][xIdx].y)
        if round(diffYokeDims, 12) != 0:
            print(f'flag - yoke: {diffYokeDims}')
            spatialDomainFlag = True
        # Check Vacuum Upper
        diffVacUpperDims = self.vac - (self.matrix[-1][xIdx].y + self.matrix[-1][xIdx].ly - self.matrix[self.yIndexesVacUpper[0]][xIdx].y)
        if round(diffVacUpperDims, 12) != 0:
            print(f'flag - vacuum upper: {diffVacUpperDims}')
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
        else:
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

            self.xCenter = self.x + self.lx/2
            self.yCenter = self.y + self.ly/2

            # Node properties
            self.ur = 0.0
            self.sigma = 0.0
            self.material = ''
            self.colour = ''

            # Potential
            self.Yk = np.cdouble(0.0)

            # Thrust
            self.Fx, self.Fy, self.F = np.zeros(3, dtype=np.cdouble)

            # B field
            self.Bx, self.By, self.B, self.BxLower, self.ByLower, self.B_Lower = np.zeros(6, dtype=np.cdouble)

            # Flux
            self.phiXp, self.phiXn, self.phiYp, self.phiYn, self.phiX, self.phiY, self.phi = np.zeros(7, dtype=np.cdouble)
            self.phiError = np.cdouble(0.0)

            # Current
            self.Iph = np.cdouble(0.0)  # phase

            # MMF
            self.MMF = np.cdouble(0.0)  # AmpereTurns

            # Reluctance
            self.Rx, self.Ry, self.R = np.zeros(3, dtype=np.float64)

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

    def drawNode(self, canvasSpacing, overRideColour, c, nodeWidth):

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

        c.create_rectangle(x_old, y_old, x_new, y_new, width=nodeWidth, fill=fillColour)

    def getReluctance(self):

        ResX = self.lx / (2 * uo * self.ur * self.Szy)
        ResY = self.ly / (2 * uo * self.ur * self.Sxz)

        return ResX, ResY


class Region(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for key in kwargs:
                if key in ['an', 'bn']:
                    valList = [rebuildPlex(index) for index in kwargs[key]]
                    valArray = np.array(valList)
                    self.__dict__[key] = valArray
                else:
                    self.__dict__[key] = kwargs[key]
        else:
            self.index = kwargs['index']
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
