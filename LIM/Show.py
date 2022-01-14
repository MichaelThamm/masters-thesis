from LIM.Compute import *
from tkinter import *
import matplotlib as mpl
import matplotlib.pyplot as plt


# class GUI:
#     def __init__(self):


# noinspection PyUnresolvedReferences
def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)

    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))

    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def myColourNumber(fieldScale, val):

    if val in [np.inf, -np.inf]:
        result = val
    else:
        result = min(fieldScale, key=lambda x: abs(x - val))

    return result


def determineColour(grid, gridInfo, iI, iJ, field):

    fieldType, iFieldsScale, iStoColours, iPosInf, iNegInf = field

    if str(type(grid[iI, iJ].__dict__[fieldType])).split("'")[1] in gridInfo['complexTypeList']:
        myNumber = myColourNumber(iFieldsScale, grid[iI, iJ].__dict__[fieldType].real)
    else:
        myNumber = myColourNumber(iFieldsScale, grid[iI, iJ].__dict__[fieldType])

    # noinspection PyUnboundLocalVariable
    colorScaleIndex = np.where(iFieldsScale == myNumber)
    if colorScaleIndex[0][0] == len(iStoColours):
        print('OH MY LAWD ITS A FIRE')
        colorScaleIndex[0][0] = len(iStoColours) - 1
    oOverRideColour = iStoColours[colorScaleIndex[0][0]]

    return oOverRideColour


def minMaxField(info, grid, attName, filtered, showFilter):

    iFilteredRows, iFilteredRowCols = filtered

    if showFilter:
        tFiltered = [[grid[x.yIndex, x.xIndex] for x in y if x.yIndex in iFilteredRows or (x.yIndex, x.xIndex) in iFilteredRowCols] for y in grid]

    else:
        tFiltered = grid

    tFilteredNoEmpties = list(filter(lambda x: True if x != [] else False, tFiltered))

    maxScale = max(map(max, [[x.__dict__[attName].real if str(type(x.__dict__[attName])).split("'")[1] in info['complexTypeList'] else x.__dict__[attName] for x in y] for y in tFilteredNoEmpties]))
    minScale = min(map(min, [[x.__dict__[attName].real if str(type(x.__dict__[attName])).split("'")[1] in info['complexTypeList'] else x.__dict__[attName] for x in y] for y in tFilteredNoEmpties]))

    return minScale.real, maxScale.real


def combineFilterList(pixelsPer, unfilteredRows, unfilteredRowCols):

    ppH, ppL = pixelsPer
    oFilteredRows = []
    oFilteredRowCols = []

    # List of row indexes for keeping rows from 0 to ppH
    for a in np.arange(len(unfilteredRows)):
        oFilteredRows += unfilteredRows[a]

    # List of (row, col) indexes for keeping nodes in mesh
    for a in np.arange(len(unfilteredRowCols)):
        for y in np.arange(ppH):
            for x in np.arange(ppL):
                if y in unfilteredRowCols[a][0] and x in unfilteredRowCols[a][1]:
                    oFilteredRowCols += [(y, x)]

    return oFilteredRows, oFilteredRowCols


def genCanvasMatrix(matrix, pixelSize, canvasRowRegions, bShowA=False, bShowB=False):

    colourSet = ['#DAF7A6', '#FFC300', '#FF5733', '#C70039', '#900C3F', '#581845']
    colourMem = 'orange'
    iterColourSet = iter(colourSet)
    oY_new = 0

    for i in range(len(matrix)):

        oY_old = oY_new
        oY_new = oY_old + pixelSize
        oX_new = 0

        if i in canvasRowRegions:
            colourMem = next(iterColourSet)

        if bShowA:

            # If iMatrix is 2D
            for j in range(len(matrix[i])):
                oX_old = oX_new
                oX_new = oX_old + pixelSize

                if matrix[i, j] == 0:
                    oFillColour = 'empty'

                else:
                    oFillColour = colourMem

                yield oX_old, oY_old, oX_new, oY_new, oFillColour

        if bShowB:

            # If iMatrix is 1D
            oX_old = 0
            oX_new = pixelSize * 10

            if matrix[i] == 0:
                oFillColour = 'empty'

            else:
                oFillColour = colourMem

            yield oX_old, oY_old, oX_new, oY_new, oFillColour


def visualizeMatrix(dims, model, bShowA=False, bShowB=False):

    if bShowA and bShowB:
        print('Please init visualizeMatrix function for matrix A and B separately')
        return

    elif not bShowA and not bShowB:
        print('Neither A or B matrix was chosen to be visualized')
        return

    zoomFactor = 3

    matrixGrid = Tk()
    matrixGrid.title('Matrix Visualization')
    frame = Frame(matrixGrid, height=dims[0] // 2, width=dims[1] // 2)
    frame.pack(expand=True, fill=BOTH)
    mGrid = Canvas(frame, height=dims[0] // 2, width=dims[1] // 2, bg='gray30', highlightthickness=0,
                   scrollregion=(0, 0, int(dims[1] * 2 * zoomFactor), int(dims[0] * 2 * zoomFactor)))

    hbar = Scrollbar(frame, orient=HORIZONTAL)
    hbar.pack(side=BOTTOM, fill=X)
    hbar.config(command=mGrid.xview)
    vbar = Scrollbar(frame, orient=VERTICAL)
    vbar.pack(side=RIGHT, fill=Y)
    vbar.config(command=mGrid.yview)

    mGrid.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)

    if bShowA:
        # Grid row region lines
        for val in model.canvasRowRegIdxs:
            mGrid.create_line(0, val * zoomFactor, dims[1] * zoomFactor, val * zoomFactor, fill='blue',
                              dash=(4, 2))  # Horizontal Lines

        # Grid col region lines
        for val in model.canvasColRegIdxs:
            mGrid.create_line(val * zoomFactor, 0, val * zoomFactor, dims[0] * zoomFactor, fill='blue',
                              dash=(4, 2))  # Vertical Lines

        # Grid MEC-HM boundary region lines
        for val in model.mecHmCanvasRegIdxs:
            mGrid.create_line(0, val * zoomFactor, dims[1] * zoomFactor, val * zoomFactor, fill='red',
                              dash=(4, 2))  # Horizontal Lines
            mGrid.create_line(val * zoomFactor, 0, val * zoomFactor, dims[1] * zoomFactor, fill='red',
                              dash=(4, 2))  # Vertical Lines

        matrixCanvasGenerator = genCanvasMatrix(model.matrixA, zoomFactor, model.canvasRowRegIdxs, bShowA=bShowA, bShowB=bShowB)

    elif bShowB:
        # Grid row region lines
        for val in model.canvasRowRegIdxs:
            mGrid.create_line(0, val * zoomFactor, dims[1] * zoomFactor, val * zoomFactor, fill='blue',
                              dash=(4, 2))  # Horizontal Lines

        # Grid MEC-HM boundary region lines
        for val in model.mecHmCanvasRegIdxs:
            mGrid.create_line(0, val * zoomFactor, dims[1] * zoomFactor, val * zoomFactor, fill='red',
                              dash=(4, 2))  # Horizontal Lines

        yPos = zoomFactor * (model.mecHmCanvasRegIdxs[0] + model.ppAirBuffer + model.ppLeftEndTooth + model.ppSlotpitch - 1)
        mGrid.create_line(0, yPos, dims[1] * zoomFactor, yPos, fill='green',
                          dash=(4, 2))  # Horizontal Lines
        # [202, 304]
        yPos = zoomFactor * (model.mecHmCanvasRegIdxs[0] + model.ppAirBuffer + model.ppLeftEndTooth + (model.slots - 4) * (model.ppSlotpitch) + model.ppSlot + 1)
        mGrid.create_line(0, yPos, dims[1] * zoomFactor, yPos, fill='green',
                          dash=(4, 2))  # Horizontal Lines

        yPos = zoomFactor * (model.mecHmCanvasRegIdxs[1] + model.ppAirBuffer + model.ppLeftEndTooth + 3 * (model.ppSlotpitch) - 1)
        mGrid.create_line(0, yPos, dims[1] * zoomFactor, yPos, fill='green',
                          dash=(4, 2))  # Horizontal Lines

        yPos = zoomFactor * (model.mecHmCanvasRegIdxs[1] + model.ppAirBuffer + model.ppLeftEndTooth + (model.slots - 2) * (model.ppSlotpitch) + model.ppSlot + 1)
        mGrid.create_line(0, yPos, dims[1] * zoomFactor, yPos, fill='green',
                          dash=(4, 2))  # Horizontal Lines

        matrixCanvasGenerator = genCanvasMatrix(model.matrixB, zoomFactor, model.canvasRowRegIdxs, bShowA=bShowA, bShowB=bShowB)

    else:
        matrixCanvasGenerator = None

    # Colour nodes that have a non-zero value
    for i in matrixCanvasGenerator:
        x_old, y_old, x_new, y_new, fillColour = i
        if fillColour == 'empty':
            pass
        else:
            mGrid.create_rectangle(x_old, y_old, x_new, y_new, width=0, fill=fillColour)

    mGrid.pack(side=LEFT, expand=True, fill=BOTH)
    mGrid.mainloop()


def showModel(gridInfo, gridMatrix, model, fieldType, showGrid, showFields, showFilter, showMatrix, numColours, dims):

    # TODO Why would we pass in both gridInfo and gridMatrix? Doesn't that defeat the purpose of dumping to json
    # Create the grid canvas to display the grid mesh
    if showGrid:
        rootGrid = Tk()
        cGrid: Canvas = Canvas(rootGrid, height=dims[0], width=dims[1], bg='gray30')
        cGrid.pack()
        i, j = 0, 0
        while i < gridMatrix.shape[0]:
            while j < gridMatrix.shape[1]:
                gridMatrix[i, j].drawNode(canvasSpacing=gridInfo['Cspacing'], overRideColour=False, c=cGrid, nodeWidth=1)
                j += 1
            j = 0
            i += 1
        rootGrid.mainloop()

    # Color nodes based on node values
    if showFields or showFilter:

        # This is for troubleshooting the distribution of flux error in the mesh
        testReal = np.array([[gridMatrix[i, j].phiError.real for j in np.arange(gridMatrix.shape[1])] for i in np.arange(gridMatrix.shape[0])])
        testImag = np.array([[gridMatrix[i, j].phiError.imag for j in np.arange(gridMatrix.shape[1])] for i in np.arange(gridMatrix.shape[0])])

        sizeArrayReal = np.array([[0 if j == 0 else 20 for j in i] for i in testReal])
        sizeArrayImag = np.array([[0 if j == 0 else 20 for j in i] for i in testImag])

        l1 = gridInfo['ppL'] * gridInfo['ppVacuumLower']
        l2 = l1 + gridInfo['ppL']
        l3 = l2 + gridInfo['ppL'] * (gridInfo['ppHeight'] - 2)
        l4 = l3 + gridInfo['ppL']
        xcoords = [l1, l2, l3, l4]
        for xc in xcoords:
            plt.axvline(x=xc)

        # marker = ".", markersize = 40

        plt.scatter(np.arange(testReal.size), testReal.flatten(), sizeArrayReal.flatten())
        plt.scatter(np.arange(testImag.size), testImag.flatten(), sizeArrayImag.flatten())
        plt.show()

        rootFields = Tk()
        cFields: Canvas = Canvas(rootFields, height=dims[0], width=dims[1], bg='gray30')
        cFields.pack()

        c1 = '#FFF888'  # Yellow Positive Limit
        c2 = '#700000'  # Dark Red Negative Limit
        cPosInf = '#00FFFF'  # BabyBlue Positive Infinity
        cNegInf = '#9EFE4C'  # Green Negative Infinity
        stoColours = ["empty"]*(numColours + 1)
        for x in range(numColours + 1):
            stoColours[x] = colorFader(c1, c2, x / numColours)

        # Max and Min values for normalizing the color scales for field analysis
        keepRows = [[]]*1
        keepRowColsUnfiltered = [[]]*2

        # [row]
        # Rule 1
        keepRows[0] = gridInfo['airgapYIndexes']

        # [row, col] - Make sure to put the rules in order of ascending rows or the list wont be sorted (shouldnt matter)
        # Rule 1
        keepRowColsUnfiltered[0] = [gridInfo['yokeYIndexes'], gridInfo['toothArray'] + gridInfo['coilArray']]
        # Rule 2
        keepRowColsUnfiltered[1] = [gridInfo['lower_slotYIndexes1'] + gridInfo['upper_slotYIndexes1'], gridInfo['toothArray']]

        filteredRows, filteredRowCols = combineFilterList([gridInfo['ppH'], gridInfo['ppL']], keepRows, keepRowColsUnfiltered)

        minScale, maxScale = minMaxField(gridInfo, gridMatrix, fieldType, [filteredRows, filteredRowCols], showFilter)
        normScale = (maxScale - minScale) / (numColours - 1)

        if [minScale, maxScale, normScale] == [0, 0, 0]:

            model.writeErrorToDict(key='name',
                                   error=Error(name='emptyField',
                                               description=f'Field Analysis Error. All values are zero! Type: {fieldType}',
                                               cause=True))

        # Create fields canvas to display the selected field result on the mesh
        else:
            if minScale < 0:
                fieldsScale = np.arange(minScale, maxScale - normScale, normScale)
            else:
                fieldsScale = np.arange(minScale, maxScale + normScale, normScale)
            colorScaleIndex = np.where(fieldsScale == fieldsScale[0])

            cFields.create_text(400, 1000, font="Purisa", text=f"Debug (Max, Min): ({maxScale}, {minScale}) colour: ({stoColours[-1]}, {stoColours[colorScaleIndex[0][0]]}) Type: {fieldType}")
            print(f"Debug (Max, Min): ({maxScale}, {minScale}) colour: ({stoColours[-1]}, {stoColours[colorScaleIndex[0][0]]}) Type: {fieldType}")

            # Assigns a colour to a node based on its relative position in the range of values and the range of available colours
            i, j, k = 0, 0, 0
            while i < gridMatrix.shape[0]:
                while j < gridMatrix.shape[1]:
                    if showFilter:
                        if i in filteredRows:
                            overRideColour = determineColour(gridMatrix, gridInfo, i, j, [fieldType, fieldsScale, stoColours, cPosInf, cNegInf])

                        elif (i, j) in filteredRowCols:
                            overRideColour = determineColour(gridMatrix, gridInfo, i, j, [fieldType, fieldsScale, stoColours, cPosInf, cNegInf])

                        else:
                            overRideColour = '#000000'
                    else:
                        overRideColour = determineColour(gridMatrix, gridInfo, i, j, [fieldType, fieldsScale, stoColours, cPosInf, cNegInf])

                    gridMatrix[i, j].drawNode(canvasSpacing=gridInfo['Cspacing'], overRideColour=overRideColour, c=cFields, nodeWidth=1)
                    j += 1
                    k += 1
                j = 0
                i += 1

        rootFields.mainloop()

    if showMatrix:
        visualizeMatrix(dims, model, bShowA=True)
        visualizeMatrix(dims, model, bShowB=True)

