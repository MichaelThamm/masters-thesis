from LIM.Compute import *
from tkinter import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from CanvasModel import CanvasInFrame


# class GUI:
#     def __init__(self):


# noinspection PyUnresolvedReferences
def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)

    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))

    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


def myColourNumber(fieldScale, val):

    return val if np.isinf(val) else min(fieldScale, key=lambda x: abs(x - val))


def determineColour(jsonObject, iI, iJ, field, highlightZeroValsInField):

    fieldType, iFieldsScale, iStoColours, iPosInf, iNegInf = field

    if str(type(jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType])).split("'")[1] in jsonObject.unacceptedTypeList:
        myNumber = myColourNumber(iFieldsScale, jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType].real)
        valEqZero = True if (jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType].real == 0 and highlightZeroValsInField) else False
    else:
        myNumber = myColourNumber(iFieldsScale, jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType])
        valEqZero = True if (jsonObject.rebuiltModel.matrix[iI, iJ].__dict__[fieldType] == 0 and highlightZeroValsInField) else False

    # noinspection PyUnboundLocalVariable
    colorScaleIndex = np.where(iFieldsScale == myNumber) if not np.isinf(myNumber) else myNumber
    if valEqZero:
        oOverRideColour = matList[-1][1]
    elif np.isinf(myNumber):
        oOverRideColour = '#000000'
    else:
        oOverRideColour = iStoColours[colorScaleIndex[0][0]]

    return oOverRideColour


def minMaxField(jsonObject, attName, filtered, showFilter):

    model = jsonObject.rebuiltModel
    iFilteredRows, iFilteredRowCols = filtered

    if showFilter:
        tFiltered = [[model.matrix[x.yIndex, x.xIndex] for x in y if x.yIndex in iFilteredRows or (x.yIndex, x.xIndex) in iFilteredRowCols] for y in model.matrix]

    else:
        tFiltered = model.matrix

    tFilteredNoEmpties = list(filter(lambda x: True if x != [] else False, tFiltered))

    attrFieldList = [x.__dict__[attName].real if type(x.__dict__[attName]) in jsonObject.unacceptedTypeList else x.__dict__[attName] for y in tFilteredNoEmpties for x in y]
    filteredAttrFieldList = list(filter(lambda x: not np.isinf(x), attrFieldList))
    maxScale = max(filteredAttrFieldList)
    minScale = min(filteredAttrFieldList)

    return minScale, maxScale


def combineFilterList(pixelsPer, unfilteredRows, unfilteredRowCols):

    ppH, ppL = pixelsPer
    oFilteredRows = []
    oFilteredRowCols = []

    # List of row indexes for keeping rows from 0 to ppH
    for a in range(len(unfilteredRows)):
        oFilteredRows += unfilteredRows[a]

    # List of (row, col) indexes for keeping nodes in mesh
    for a in range(len(unfilteredRowCols)):
        for y in range(ppH):
            for x in range(ppL):
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

    zoomFactor = 5

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
        for val in model.mecCanvasRegIdxs:
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
        for val in model.mecCanvasRegIdxs:
            mGrid.create_line(0, val * zoomFactor, dims[1] * zoomFactor, val * zoomFactor, fill='red',
                              dash=(4, 2))  # Horizontal Lines

        constantOffset1 = model.mecCanvasRegIdxs[0] + model.ppL * (model.ppYoke - 1) + model.ppAirBuffer + model.ppLeftEndTooth + model.ppSlotpitch - 1
        # TODO The green lines are dependant on the mesh size (ppSlot and ppTooth)
        #  since the source is eastMMFNum / eastRelDenom - westMMFNum / westRelDenom this is what determines the B source
        #  if I change the mesh density these green lines will not match up: different results for ppSlot and ppTooth
        #  less than 3
        constantOffset2 = constantOffset1 + 2  # This number should always be 2 if model.ppSlot > 2
        constantOffset3 = constantOffset2 + (model.ppSlot - 2)
        constantOffset4 = constantOffset3 + 4
        yPos = zoomFactor * constantOffset1
        mGrid.create_line(0, yPos, dims[1] * zoomFactor, yPos, fill='green',
                          dash=(4, 2))  # Horizontal Lines
        yPos = zoomFactor * constantOffset2
        mGrid.create_line(0, yPos, dims[1] * zoomFactor, yPos, fill='green',
                          dash=(4, 2))  # Horizontal Lines

        yPos = zoomFactor * constantOffset3
        mGrid.create_line(0, yPos, dims[1] * zoomFactor, yPos, fill='green',
                          dash=(4, 2))  # Horizontal Lines

        yPos = zoomFactor * constantOffset4
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


def showModel(jsonObject, fieldType, showGrid, showFields, showFilter, showMatrix, showZeros, numColours, dims, invertY):

    invertCoeff = -1 if invertY else 1

    # Create the grid canvas to display the grid mesh
    if showGrid:

        cGrid = CanvasInFrame(height=dims[0], width=dims[1], bg='gray30')
        i, j = 0, 0
        while i < jsonObject.rebuiltModel.matrix.shape[0]:
            while j < jsonObject.rebuiltModel.matrix.shape[1]:
                jsonObject.rebuiltModel.matrix[i, j].drawNode(canvasSpacing=jsonObject.rebuiltModel.Cspacing,
                                                              overRideColour=False, c=cGrid.canvas, nodeWidth=1)
                j += 1
            j = 0
            i += 1

        cGrid.canvas.scale("all", 0, 0, 1, invertCoeff)
        cGrid.canvas.configure(scrollregion=cGrid.canvas.bbox("all"))
        cGrid.root.mainloop()

    # Color nodes based on node values
    if showFields or showFilter:

        cFields = CanvasInFrame(height=dims[0], width=dims[1], bg='gray30')

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
        keepRows[0] = jsonObject.rebuiltModel.yIndexesAirgap

        # [row, col] - Make sure to put the rules in order of ascending rows or the list wont be sorted (shouldnt matter)
        # Rule 1
        keepRowColsUnfiltered[0] = [jsonObject.rebuiltModel.yIndexesYoke, jsonObject.rebuiltModel.toothArray + jsonObject.rebuiltModel.coilArray]
        # Rule 2
        keepRowColsUnfiltered[1] = [jsonObject.rebuiltModel.yIndexesLowerSlot + jsonObject.rebuiltModel.yIndexesUpperSlot, jsonObject.rebuiltModel.toothArray]

        filteredRows, filteredRowCols = combineFilterList([jsonObject.rebuiltModel.ppH, jsonObject.rebuiltModel.ppL], keepRows, keepRowColsUnfiltered)

        minScale, maxScale = minMaxField(jsonObject, fieldType, [filteredRows, filteredRowCols], showFilter)
        normScale = (maxScale - minScale) / (numColours - 1)

        if [minScale, maxScale, normScale] == [0, 0, 0]:

            jsonObject.rebuiltModel.writeErrorToDict(key='name',
                                   error=Error.buildFromScratch(name='emptyField',
                                                                description=f'Field Analysis Error. All values are zero! Type: {fieldType}',
                                                                cause=True))

        # Create fields canvas to display the selected field result on the mesh
        else:
            fieldsScale = np.arange(minScale, maxScale + normScale, normScale)
            colorScaleIndex = np.where(fieldsScale == fieldsScale[0])

            cFields.canvas.create_text(400, 1000, font="Purisa", text=f"Debug (Max, Min): ({maxScale}, {minScale}) colour: ({stoColours[-1]}, {stoColours[colorScaleIndex[0][0]]}) Type: {fieldType}")
            print(f"Debug (Max, Min): ({maxScale}, {minScale}) colour: ({stoColours[-1]}, {stoColours[colorScaleIndex[0][0]]}) Type: {fieldType}")

            # All drawing is done at the bottom of the node
            def horCoreBoundary(tuple_in):
                if tuple_in[0] == jsonObject.rebuiltModel.yIndexesYoke[0] and tuple_in[1] in jsonObject.rebuiltModel.toothArray + jsonObject.rebuiltModel.coilArray:
                    return True
                elif tuple_in[0] == jsonObject.rebuiltModel.yIndexesUpperSlot[0] and tuple_in[1] in jsonObject.rebuiltModel.coilArray:
                    return True
                elif tuple_in[0] == jsonObject.rebuiltModel.yIndexesMEC[-1] + 1 and tuple_in[1] in jsonObject.rebuiltModel.toothArray:
                    return True
                else:
                    return False

            # All drawing is done to the right of the node
            def vertCoreBoundary(tuple_in):
                if tuple_in[0] in jsonObject.rebuiltModel.yIndexesMEC:

                    # Left end tooth
                    if tuple_in[1] in [jsonObject.rebuiltModel.toothArray[0] - 1] + [jsonObject.rebuiltModel.toothArray[jsonObject.rebuiltModel.ppLeftEndTooth-1]]:
                        return True
                    # Right end tooth
                    elif tuple_in[1] in [jsonObject.rebuiltModel.toothArray[-jsonObject.rebuiltModel.ppRightEndTooth-1]] + [jsonObject.rebuiltModel.toothArray[-1]]:
                        return True
                    # Remaining teeth left edge
                    elif tuple_in[0] not in jsonObject.rebuiltModel.yIndexesYoke and tuple_in[1] in jsonObject.rebuiltModel.toothArray[jsonObject.rebuiltModel.ppLeftEndTooth:-jsonObject.rebuiltModel.ppRightEndTooth:jsonObject.rebuiltModel.ppSlotpitch]:
                        return True
                    # Remaining teeth right edge
                    elif tuple_in[0] not in jsonObject.rebuiltModel.yIndexesYoke and tuple_in[1] in jsonObject.rebuiltModel.toothArray[jsonObject.rebuiltModel.ppLeftEndTooth+jsonObject.rebuiltModel.ppTooth:-jsonObject.rebuiltModel.ppRightEndTooth:jsonObject.rebuiltModel.ppSlotpitch]:
                        return True
                    else:
                        return False

                else:
                    return False

            nodeIndexes = [(node.yIndex, node.xIndex) for row in jsonObject.rebuiltModel.matrix for node in row]
            horCoreBoundaryIdxs = filter(horCoreBoundary, nodeIndexes)
            vertCoreBoundaryIdxs = filter(vertCoreBoundary, nodeIndexes)

            # Assigns a colour to a node based on its relative position in the range of values and the range of available colours
            i, j, k = 0, 0, 0
            while i < jsonObject.rebuiltModel.matrix.shape[0]:
                while j < jsonObject.rebuiltModel.matrix.shape[1]:
                    if showFilter:
                        if i in filteredRows or (i, j) in filteredRowCols:
                            overRideColour = determineColour(jsonObject, i, j,
                                                             [fieldType, fieldsScale, stoColours, cPosInf, cNegInf],
                                                             highlightZeroValsInField=showZeros)

                        else:
                            overRideColour = '#000000'
                    else:
                        overRideColour = determineColour(jsonObject, i, j,
                                                         [fieldType, fieldsScale, stoColours, cPosInf, cNegInf],
                                                         highlightZeroValsInField=showZeros)

                    jsonObject.rebuiltModel.matrix[i, j].drawNode(canvasSpacing=jsonObject.rebuiltModel.Cspacing,
                                                                  overRideColour=overRideColour, c=cFields.canvas,
                                                                  nodeWidth=1)
                    j += 1
                    k += 1
                j = 0
                i += 1

        cFields.canvas.scale("all", 0, 0, 1, invertCoeff)
        cFields.canvas.configure(scrollregion=cFields.canvas.bbox("all"))
        cFields.root.mainloop()

    if showMatrix:
        visualizeMatrix(dims, jsonObject.rebuiltModel, bShowA=True)
        visualizeMatrix(dims, jsonObject.rebuiltModel, bShowB=True)

