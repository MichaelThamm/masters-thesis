import json
import numpy as np
import matplotlib.pyplot as plt
from LIM.LIM_Platypus import construct, LIM_Show, LimMotor


def plotPointsAlongX(iGridInfo, iGridMatrix, iY):

    evenOdd = 'even' if (iY - iGridInfo['ppHeight'] - iGridInfo['ppVacuumLower']) % 2 == 0 else 'odd'

    # X axis array
    xCenterPosList = np.array([node.xCenter for node in iGridMatrix[iY, :]])
    dataArray = np.zeros((4, len(xCenterPosList)), dtype=np.cdouble)
    dataArray[0] = xCenterPosList

    #  Y axis array
    if evenOdd == 'even':  # even

        yBxLowerList = np.array([iGridMatrix[iY, j].BxLower for j in np.arange(iGridInfo['ppL'])], dtype=np.cdouble)
        yByLowerList = np.array([iGridMatrix[iY, j].ByLower for j in np.arange(iGridInfo['ppL'])], dtype=np.cdouble)
        yB_LowerList = np.array([iGridMatrix[iY, j].B_Lower for j in np.arange(iGridInfo['ppL'])], dtype=np.cdouble)

        dataArray[1] = yBxLowerList
        dataArray[2] = yByLowerList
        dataArray[3] = yB_LowerList

    else:  # odd
        yB_CenterList = np.array([iGridMatrix[iY, j].B for j in np.arange(iGridInfo['ppL'])], dtype=np.cdouble)
        yBxCenterList = np.array([iGridMatrix[iY, j].Bx for j in np.arange(iGridInfo['ppL'])], dtype=np.cdouble)
        yByCenterList = np.array([iGridMatrix[iY, j].By for j in np.arange(iGridInfo['ppL'])], dtype=np.cdouble)

        dataArray[1] = yBxCenterList
        dataArray[2] = yByCenterList
        dataArray[3] = yB_CenterList

    # Sort data based on x axis
    sortedArray = dataArray[:, dataArray[0].argsort()]
    xSorted = np.array([a.real for a in sortedArray[0]], dtype=np.float64)

    testReal = np.array([j.real for j in dataArray[1]], dtype=np.float64)
    plt.scatter(xSorted, testReal.flatten())
    plt.plot(xSorted, testReal.flatten())
    plt.xlabel('Position [m]')
    plt.ylabel('Bx [T]]')
    plt.title('Bx field in airgap')
    plt.show()

    testReal = np.array([j.real for j in dataArray[2]])
    plt.scatter(xSorted, testReal.flatten())
    plt.plot(xSorted, testReal.flatten())
    plt.xlabel('Position [m]')
    plt.ylabel('By [T]]')
    plt.title('By field in airgap')
    plt.show()

    testReal = np.array([j.real for j in dataArray[3]])
    plt.scatter(xSorted, testReal.flatten())
    plt.plot(xSorted, testReal.flatten())
    plt.xlabel('Position [m]')
    plt.ylabel('|B| [T]]')
    plt.title('B field in airgap')
    plt.show()


def main():
    errorList = []

    with open("../Output/StoredSolutionData.json") as StoredSolutionData:
        dictionary = json.load(StoredSolutionData)
    tempMotor = LimMotor(iSlots=dictionary['info']['slots'], iPoles=dictionary['info']['poles'], iL=dictionary['info']['length'])

    gridInfo, rebuilt__Grid_matrix = construct(iDict=dictionary, iArrayShape=[dictionary['info']['ppH'], dictionary['info']['ppL']], iDesign=tempMotor)

    centerAirgapIndex_Y = gridInfo['ppVacuumLower'] + gridInfo['ppHeight'] + gridInfo['ppAirgap'] // 2

    plotPointsAlongX(gridInfo, rebuilt__Grid_matrix, centerAirgapIndex_Y)

    # iDims (height x width): BenQ = 1440 x 2560, ViewSonic = 1080 x 1920
    # TODO This will not work since the gridMatrix A and some of the results from UpdateGrid are not stored in the JSON object. The solution is to either store them in the JSON object (annoying, but good for future expansion)
    # TODO Or we can add a switch statement in LIM_Show to signify if we are using it in ShowFromJSON or in Platypus
    errorList = LIM_Show(gridInfo, rebuilt__Grid_matrix, iFieldType='MMF', iShowGrid=True, iShowFields=True, iShowFilter=False, iNumColours=350, errorList=errorList, iDims=[1440, 2560])

    if errorList:
        print('\n')
        print('Below are a list of warnings and errors:')
        for error in errorList:
            print(error.description)


if __name__ == '__main__':
    main()
