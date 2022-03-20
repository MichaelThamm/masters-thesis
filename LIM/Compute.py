from LIM.Grid import *
from LIM.SlotPoleCalculation import np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from functools import lru_cache


# This performs the lower-upper decomposition of A to solve for x in Ax = B
# A must be a square matrix
# @njit("float64[:](float64[:, :], float64[:])", cache=True)
class Model(Grid):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for attr_key in kwargs:
                self.__dict__[attr_key] = kwargs[attr_key]
        else:
            super().__init__(kwargs)

            self.writeErrorToDict(key='name',
                                  error=Error.buildFromScratch(name='meshDensityDiscrepancy',
                                                               description="ERROR - The last slot has a different mesh density than all other slots",
                                                               cause=kwargs['meshIndexes'][0][2] != kwargs['meshIndexes'][0][-3]))

            self.currColCount = 0
            self.nLoop = 0
            self.matBCount = 0

            lenUnknowns = self.hmRegionsIndex[-1]

            self.matrixA = np.zeros((lenUnknowns, lenUnknowns), dtype=np.cdouble)
            self.matrixB = np.zeros(len(self.matrixA), dtype=np.cdouble)
            self.matrixX = np.zeros(len(self.matrixA), dtype=np.cdouble)

            self.hmUnknownsList = {i: Region.buildFromScratch(index=i, type=self.hmRegions[i], an=np.zeros(len(self.n)), bn=np.zeros(len(self.n))) for i in self.hmRegions}

            # HM and MEC unknown indexes in matrix A, used for visualization
            # TODO I could make these lists more robust
            # self.canvasRowRegIdxs = [len(self.n), len(self.n) + self.mecRegionLength,
            #                                self.mecRegionLength + 2 * len(self.n), self.mecRegionLength + 4 * len(self.n),
            #                                self.mecRegionLength + 6 * len(self.n), self.mecRegionLength + 8 * len(self.n)]
            # self.canvasColRegIdxs = [len(self.n), len(self.n) + self.mecRegionLength,
            #                                self.mecRegionLength + 3 * len(self.n), self.mecRegionLength + 5 * len(self.n),
            #                                self.mecRegionLength + 7 * len(self.n), self.mecRegionLength + 8 * len(self.n)]
            # self.mecCanvasRegIdxs = [self.canvasRowRegIdxs[0] + self.ppL * i for i in range(1, self.ppHeight)]

            self.hmIdxs = list(range(self.mecRegionsIndex[0])) + list(range(self.hmRegionsIndex[4], lenUnknowns))
            self.mecIdxs = [i for i in range(len(self.matrixX)) if i not in self.hmIdxs]

            self.hmMatrixX = []
            self.mecMatrixX = []

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    def equals(self, otherObject, removedAtts):
        equality = True
        if not isinstance(otherObject, Model):
            # don't attempt to compare against unrelated types
            equality = False

        else:
            # Compare items between objects
            selfItems = list(filter(lambda x: True if x[0] not in removedAtts else False, self.__dict__.items()))
            otherItems = list(otherObject.__dict__.items())
            for cnt, entry in enumerate(selfItems):
                # if entry[0] == 'matrix':
                #     if  :
                #
                #     else:
                #         equality = False
                if type(entry[1]) == np.ndarray:
                    value = entry[1].tolist()
                else:
                    value = entry[1]

                # Check keys and values
                if entry[0] == otherItems[cnt][0] and value == otherItems[cnt][1]:
                    pass
                else:
                    equality = False

        return True if equality else False
        # return isinstance(obj, MyFoo) and obj.equalityprop == self.equalityprop

    # TODO We can comment these out for computation efficiency since the bn term is getting removed entirely and
    #  the an term is getting 0 always which matrix A is initialized as np.zeros()
    # ___Boundary Condition Methods___ #
    def __dirichlet(self, regCountOffset, yBoundary):
        # Eqn 22 in Aleksandrov2018 states that at y = -inf: bn = 0 and the coefficient for an = 0
        # bn can be taken out of the matrix eqn because it is solved.
        if yBoundary == '-inf':
            self.matrixA[self.nLoop, regCountOffset + self.currColCount] = 0.0  # an
            self.matrixA[self.nLoop, regCountOffset + self.currColCount + 1] = np.inf  # bn - Note: bn is being removed

        # Eqn 22 in Aleksandrov2018 states that at y = inf: an = 0 and the coefficient on bn = 0
        # an can be taken out of the matrix eqn because it is solved.
        elif yBoundary == 'inf':
            self.matrixA[self.nLoop, regCountOffset + self.currColCount] = np.inf  # an - Note: an is being removed
            self.matrixA[self.nLoop, regCountOffset + self.currColCount + 1] = 0.0  # bn

        else:
            print('You did not enter a valid yBoundary for Dirichlet condition')

        self.currColCount += 2
        self.nLoop += 1
        self.matBCount += 1

    def __hmHm(self, nHM, listBCInfo, RegCountOffset1, RegCountOffset2):

        hb1, ur1, urSigma1, _, ur2, urSigma2 = listBCInfo

        wn = 2 * nHM * pi / self.Tper
        lambdaN1 = self.__lambda_n(wn, urSigma1)
        lambdaN2 = self.__lambda_n(wn, urSigma2)

        # By Condition
        Ba_lower, Ba_upper, Bb_lower, Bb_upper = self.__preEqn24_2018(lambdaN1, lambdaN2, hb1)

        self.matrixA[self.nLoop, RegCountOffset1 + self.currColCount] = Ba_lower  # an Lower
        self.matrixA[self.nLoop, RegCountOffset1 + 1 + self.currColCount] = Bb_lower  # bn Lower
        self.matrixA[self.nLoop, RegCountOffset2 + self.currColCount] = Ba_upper  # an Upper
        self.matrixA[self.nLoop, RegCountOffset2 + 1 + self.currColCount] = Bb_upper  # bn Upper

        # Hx Condition
        Ha_lower, Ha_upper, Hb_lower, Hb_upper = self.__preEqn25_2018(lambdaN1, lambdaN2, ur1, ur2, hb1)

        self.matrixA[self.nLoop + 1, RegCountOffset1 + self.currColCount] = Ha_lower  # an Lower
        self.matrixA[self.nLoop + 1, RegCountOffset1 + 1 + self.currColCount] = Hb_lower  # bn Lower
        self.matrixA[self.nLoop + 1, RegCountOffset2 + self.currColCount] = Ha_upper  # an Upper
        self.matrixA[self.nLoop + 1, RegCountOffset2 + 1 + self.currColCount] = Hb_upper  # bn Upper

        self.currColCount += 2
        self.nLoop += 2
        self.matBCount += 2

    def __mecHm(self, nHM, iY, listBCInfo, hmRegCountOffset, mecRegCountOffset, removed_an, removed_bn, lowerUpper):
        hb, ur, urSigma = listBCInfo
        wn = 2 * nHM * pi / self.Tper
        lambdaN = self.__lambda_n(wn, urSigma)
        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        # TODO My math shows this to be 1 2019 paper says 2
        coeff = 2 * j_plex / (wn * self.Tper)

        if iY not in [self.yIndexesMEC[0], self.yIndexesMEC[-1]]:
            print('Please choose a valid boundary index')
            return

        row = self.matrix[iY]

        sumResEqn22Source = np.cdouble(0)
        for iX in range(self.ppL):
            lNode, rNode = self.neighbourNodes(iX)

            # By Condition
            # This is handled in the MEC region KCL equations using Eqn 21

            # Hx Condition
            # MEC related equations
            eastRelDenom = row[iX].Rx + row[rNode].Rx
            westRelDenom = row[iX].Rx + row[lNode].Rx

            # TODO This may not be correct. This affects the thrust result but not the waveform amplitude or shape
            eastMMF = row[iX].MMF + row[rNode].MMF
            westMMF = row[iX].MMF + row[lNode].MMF

            # For this section refer to the 2015 HAM paper for these 3 cases
            # __________k = k - 1__________ #
            coeffIntegral_kn = self.__eqn23Integral(wn, row[lNode].x, row[lNode].x + row[lNode].lx, row[lNode].ur, row[lNode].Szy)
            resPsi_kn = coeffIntegral_kn / westRelDenom
            resSource_kn = coeffIntegral_kn * (westMMF / westRelDenom)

            # __________k = k__________ #
            coeffIntegral_k = self.__eqn23Integral(wn, row[iX].x, row[iX].x + row[iX].lx, row[iX].ur, row[iX].Szy)
            resPsi_k = coeffIntegral_k * (1 / westRelDenom - 1 / eastRelDenom)
            resSource_k = coeffIntegral_k * (westMMF / westRelDenom + eastMMF / eastRelDenom)

            # __________k = k + 1__________ #
            coeffIntegral_kp = self.__eqn23Integral(wn, row[rNode].x, row[rNode].x + row[rNode].lx, row[rNode].ur, row[rNode].Szy)
            resPsi_kp = - coeffIntegral_kp / eastRelDenom
            resSource_kp = coeffIntegral_kp * (eastMMF / eastRelDenom)

            combinedPsi_k = time_plex * coeff * (resPsi_kn + resPsi_k + resPsi_kp)

            if lowerUpper == 'lower':
                self.matrixA[self.nLoop, mecRegCountOffset + iX] = combinedPsi_k  # Curr
            elif lowerUpper == 'upper':
                self.matrixA[self.nLoop, mecRegCountOffset + self.mecRegionLength - self.ppL + iX] = combinedPsi_k  # Curr

            sumResEqn22Source += (resSource_kn + resSource_k + resSource_kp)

        # HM related equations
        hmResA, hmResB = self.__preEqn8(ur, lambdaN, wn, hb)

        # If the neighbouring region is a Dirichlet region, an or bn should be set to np.inf which helps catch errors
        if removed_an:
            self.matrixA[self.nLoop, hmRegCountOffset + self.currColCount] = np.inf  # an
        else:
            self.matrixA[self.nLoop, hmRegCountOffset + self.currColCount] = hmResA  # an

        if removed_bn:
            self.matrixA[self.nLoop, hmRegCountOffset + 1 + self.currColCount] = np.inf  # bn
        else:
            self.matrixA[self.nLoop, hmRegCountOffset + 1 + self.currColCount] = hmResB  # bn

        self.matrixB[self.matBCount] = - coeff * sumResEqn22Source

        self.currColCount += 2
        self.nLoop += 1
        self.matBCount += 1

    def __mec(self, i, j, node, time_plex, listBCInfo,
              hmRegCountOffset1, hmRegCountOffset2, mecRegCountOffset):

        hb1, urSigma1, hb2, urSigma2 = listBCInfo

        lNode, rNode = self.neighbourNodes(j)

        northRelDenom = self.matrix[i, j].Ry + self.matrix[i + 1, j].Ry
        eastRelDenom = self.matrix[i, j].Rx + self.matrix[i, rNode].Rx
        southRelDenom = self.matrix[i, j].Ry + self.matrix[i - 1, j].Ry
        westRelDenom = self.matrix[i, j].Rx + self.matrix[i, lNode].Rx

        eastMMFNum = self.matrix[i, j].MMF + self.matrix[i, rNode].MMF
        westMMFNum = self.matrix[i, j].MMF + self.matrix[i, lNode].MMF

        # Bottom layer of the mesh
        if i == self.yIndexesMEC[0]:

            self.__setCurrColCount(0)
            # South Node
            for nHM in self.n:
                wn = 2 * nHM * pi / self.Tper
                lambdaN1 = self.__lambda_n(wn, urSigma1)
                anCoeff, bnCoeff = self.__preEqn21(lambdaN1, wn, self.matrix[i, j].x,
                                                   self.matrix[i, j].x + self.matrix[i, j].lx, hb1)

                self.matrixA[self.nLoop + node, hmRegCountOffset1 + self.currColCount] = anCoeff  # an
                self.matrixA[self.nLoop + node, hmRegCountOffset1 + 1 + self.currColCount] = bnCoeff  # bn - Note: bn is being removed
                self.currColCount += 2

            # North Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node + self.ppL] = - time_plex / northRelDenom

            # Current Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / northRelDenom)

        # Top layer of the mesh
        elif i == self.yIndexesMEC[-1]:

            self.__setCurrColCount(0)
            # North Node
            for nHM in self.n:
                wn = 2 * nHM * pi / self.Tper
                lambdaN2 = self.__lambda_n(wn, urSigma2)
                anCoeff, bnCoeff = self.__preEqn21(lambdaN2, wn, self.matrix[i, j].x,
                                                   self.matrix[i, j].x + self.matrix[i, j].lx, hb2)

                self.matrixA[self.nLoop + node, hmRegCountOffset2 + self.currColCount] = - anCoeff  # an
                self.matrixA[self.nLoop + node, hmRegCountOffset2 + 1 + self.currColCount] = - bnCoeff  # bn
                self.currColCount += 2

            # South Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node - self.ppL] = - time_plex / southRelDenom

            # Current Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / southRelDenom)

        else:
            # North Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node + self.ppL] = - time_plex / northRelDenom

            # South Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node - self.ppL] = - time_plex / southRelDenom

            # Current Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / northRelDenom + 1 / southRelDenom)

        # West Edge
        if node % self.ppL == 0:
            # West Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node + self.ppL - 1] = - time_plex / westRelDenom
            # East Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node + 1] = - time_plex / eastRelDenom

        # East Edge
        elif (node + 1) % self.ppL == 0:
            # West Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node - 1] = - time_plex / westRelDenom
            # East Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node - self.ppL + 1] = - time_plex / eastRelDenom

        else:
            # West Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node - 1] = - time_plex / westRelDenom
            # East Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node + 1] = - time_plex / eastRelDenom

        # Result
        self.matrixB[self.matBCount + node] = eastMMFNum / eastRelDenom - westMMFNum / westRelDenom

    def __preEqn24_2018(self, lam_lower, lam_upper, y):

        resA_lower = cmath.exp(lam_lower * y)
        resA_upper = - cmath.exp(lam_upper * y)
        resB_lower = cmath.exp(-lam_lower * y)
        resB_upper = - cmath.exp(-lam_upper * y)

        return resA_lower, resA_upper, resB_lower, resB_upper

    def __preEqn25_2018(self, lam_lower, lam_upper, ur_lower, ur_upper, y):

        resA_lower = (lam_lower / ur_lower) * cmath.exp(lam_lower * y)
        resA_upper = - (lam_upper / ur_upper) * cmath.exp(lam_upper * y)
        resB_lower = - (lam_lower / ur_lower) * cmath.exp(-lam_lower * y)
        resB_upper = (lam_upper / ur_upper) * cmath.exp(-lam_upper * y)

        return resA_lower, resA_upper, resB_lower, resB_upper

    def postMECAvgB(self, fluxN, fluxP, S_xyz):

        res = (fluxN + fluxP) / (2 * S_xyz)

        return res

    def __preEqn8(self, ur, lam, wn, y):

        coeff = cmath.exp(j_plex * (2 * pi * self.f * self.t + wn * self.vel * self.t))
        resA = - (lam / ur) * coeff * cmath.exp(lam * y)
        resB = (lam / ur) * coeff * cmath.exp(-lam * y)

        return resA, resB

    def __postEqn8(self, lam, wn, x, y, an, bn):

        res1 = lam * (an * cmath.exp(lam * y) - bn * cmath.exp(-lam * y))
        res2 = cmath.exp(j_plex * wn * x) * cmath.exp(j_plex * (2 * pi * self.f * self.t + wn * self.vel * self.t))
        res = res1 * res2

        return res

    def __postEqn9(self, lam, wn, x, y, an, bn):

        res1 = wn * (an * cmath.exp(lam * y) + bn * cmath.exp(-lam * y))
        res2 = cmath.exp(j_plex * wn * x) * cmath.exp(j_plex * (2 * pi * self.f * self.t + wn * self.vel * self.t))
        res = - j_plex * res1 * res2

        return res

    # @njit('float64(float64, float64, float64, float64, float64, float64)', cache=True)
    def __postEqn14to15(self, destpot, startpot, startMMF, destMMF, destRel, startRel):

        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        result = (time_plex * (destpot - startpot) + destMMF + startMMF) / (destRel + startRel)

        return result

    # @njit('float64(float64, float64, float64, float64, float64, float64)', cache=True)
    def __postEqn16to17(self, destpot, startpot, destRel, startRel):

        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        result = time_plex * (destpot - startpot) / (destRel + startRel)

        return result

    # @njit('float64(float64, float64, float64, float64, float64, float64)', cache=True)
    def __preEqn21(self, lam, wn, Xl, Xr, y):

        res1 = cmath.exp(j_plex * wn * Xl) - cmath.exp(j_plex * wn * Xr)
        res2 = cmath.exp(j_plex * 2 * pi * self.f * self.t)

        aExp = cmath.exp(lam * y)
        bExp = cmath.exp(-lam * y)

        resA = self.D * res1 * res2 * aExp
        resB = self.D * res1 * res2 * bExp

        return resA, resB

    # @njit('float64(float64, float64, float64, float64, float64, float64)', cache=True)
    def __postEqn21(self, lam, wn, Xl, Xr, y, an, bn):

        res1 = an * cmath.exp(lam * y) + bn * cmath.exp(-lam * y)
        res2 = cmath.exp(j_plex * wn * Xl) - cmath.exp(j_plex * wn * Xr)
        res3 = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        res = self.D * res1 * res2 * res3

        return res

    def __eqn23Integral(self, wn, Xl, Xr, ur, Szy):

        coeff = 1 / (2 * ur * Szy)
        res = coeff * (cmath.exp(-j_plex * wn * (Xr - self.vel * self.t)) -
                       cmath.exp(-j_plex * wn * (Xl - self.vel * self.t)))

        return res

    # ___Manipulate Matrix Methods___ #
    def __checkForErrors(self):

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='nLoop',
                                                           description='ERROR - nLoop',
                                                           cause=self.nLoop != self.matrixA.shape[0]))

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='matBCount',
                                                           description='ERROR - matBCount',
                                                           cause=self.matBCount != self.matrixB.size))

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='regCount',
                                                           description='ERROR - regCount',
                                                           cause=self.hmRegionsIndex[-1] != self.matrixA.shape[1]))

    def neighbourNodes(self, j):
        # boundary 1 to the left
        if j == 0:
            lNode = self.ppL - 1
            rNode = j + 1
        # boundary 1 to the right
        elif j == self.ppL - 1:
            lNode = j - 1
            rNode = 0
        else:
            lNode = j - 1
            rNode = j + 1

        return lNode, rNode

    # TODO We can cache functions like this for time improvement
    def __boundaryInfo(self, iY1, iY2, boundaryType):

        if boundaryType == 'mec':
            hb1 = self.matrix[iY1, 0].y
            ur1 = self.matrix[iY1 - 1, 0].ur
            sigma1 = self.matrix[iY1 - 1, 0].sigma
            urSigma1 = ur1 * sigma1
            hb2 = self.matrix[iY2+1, 0].y
            ur2 = self.matrix[iY2+1, 0].ur
            sigma2 = self.matrix[iY2+1, 0].sigma
            urSigma2 = ur2 * sigma2

            return [hb1, urSigma1, hb2, urSigma2]

        elif boundaryType == 'hmHM' and not iY2:
            hb1 = self.matrix[iY1, 0].y
            ur1 = self.matrix[iY1 - 1, 0].ur
            sigma1 = self.matrix[iY1 - 1, 0].sigma
            urSigma1 = ur1 * sigma1
            ur2 = self.matrix[iY1, 0].ur
            sigma2 = self.matrix[iY1, 0].sigma
            urSigma2 = ur2 * sigma2

            return [hb1, ur1, urSigma1, None, ur2, urSigma2]

        elif boundaryType == 'hmMEC' and not iY2:
            hb = self.matrix[iY1, 0].y
            ur = self.matrix[iY1 - 1, 0].ur
            sigma = self.matrix[iY1 - 1, 0].sigma
            urSigma = ur * sigma

            return [hb, ur, urSigma]

        elif boundaryType == 'mecHM' and not iY2:
            hb = self.matrix[iY1 + 1, 0].y
            ur = self.matrix[iY1 + 1, 0].ur
            sigma = self.matrix[iY1 + 1, 0].sigma
            urSigma = ur * sigma

            return [hb, ur, urSigma]

        else:
            print('An incorrect boundary type was chosen')

    def __reduceMatrix(self, iRowDelete, iColDelete):

        preDeleteA = self.matrixA.shape

        # Remove rows and columns described in HAM - Section V. CALCULATING THE COEFFICIENTS
        print('before mod: ', self.matrixA.shape)
        self.matrixA = np.delete(self.matrixA, iRowDelete, 0)
        self.matrixB = np.delete(self.matrixB, iRowDelete, 0)
        self.matrixA = np.delete(self.matrixA, iColDelete, 1)
        print('after mod: ', self.matrixA.shape)

        # Identify which rows and columns have np.inf in them
        rows, cols = np.where(self.matrixA == np.inf)
        print('rows with infinity: ', rows, 'cols with infinity: ', cols)

        # Check rows
        zeroRowsList = np.all(self.matrixA == 0, axis=1)
        removeRowIdx = np.where(zeroRowsList)
        print('Zero Row Indexes: ', removeRowIdx[0])

        # Check Columns
        zeroColsList = np.all(self.matrixA == 0, axis=0)
        removeColIdx = np.where(zeroColsList)
        print('Zero Column Indexes: ', removeColIdx[0])

        postDeleteA = self.matrixA.shape

        if preDeleteA != postDeleteA:
            print(f"Matrix A had {preDeleteA[0] - postDeleteA[0]} rows removed and {preDeleteA[1] - postDeleteA[1]} "
                  f"columns removed with a {round(100 - postDeleteA[0] * postDeleteA[1] / preDeleteA[0] / preDeleteA[1] * 100, 2)} percent decrease in size")
            print('New A size: ', self.matrixA.shape, self.matrixA.size, 'New B size: ', self.matrixB.size)

        if self.matrixA.shape[0] != self.matrixB.size:
            print("Error - The modified A and B matrices do not match")

    def __updateLists(self, idx):
        # This method updates the hm and mec index lists after some indexes were removed in __reduceMatrix method

        dirichletIdxs = list(range(len(self.n), 2 * len(self.n))) +\
                        list(range(self.hmRegionsIndex[-1] - len(self.n), self.hmRegionsIndex[-1]))
        # TODO This can be improved with map method rather than list comprehension
        self.hmIdxs = [index for index in self.hmIdxs if index not in dirichletIdxs]
        self.__shiftHmIdxList(idx)
        self.mecIdxs = [i for i in range(len(self.matrixX)) if i not in self.hmIdxs]

    def __shiftHmIdxList(self, idx):
        self.hmIdxs = self.hmIdxs[:len(self.n)] + [i - idx for i in self.hmIdxs[len(self.n):]]

    def __linalg_lu(self, tolerance):

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='linalg deepcopy',
                                                           description="ERROR - Matrix A and B should be ndarray so matrix deep copy is not performed",
                                                           cause=type(self.matrixA) != np.ndarray or type(self.matrixB) != np.ndarray))

        if self.matrixA.shape[0] == self.matrixA.shape[1]:
            lu, piv = lu_factor(self.matrixA)
            resX = lu_solve((lu, piv), self.matrixB)
            remainder = self.matrixA @ resX - self.matrixB
            print(f'This was the max error seen in the solution for x: {max(remainder)} vs min: {min(remainder)}')
            testPass = np.allclose(remainder, np.zeros((len(self.matrixA),)), atol=tolerance)
            print(f'LU Decomp test: {testPass}, if True then x is a solution of Ax = iB with a tolerance of {tolerance}')
            if testPass:
                return resX, max(remainder)
            else:
                print('LU Decomp test failed')
                return
        else:
            print('Error - A is not a square matrix')
            return

    def __setCurrColCount(self, value):
        self.currColCount = value

    def __lambda_n(self, wn, urSigma):
        return cmath.sqrt(wn ** 2 + j_plex * uo * urSigma * (2 * pi * self.f + wn * self.vel))

    def __genForces(self, urSigma, iY):

        Cnt = 0
        for nHM in self.n:
            gIdx = list(self.hmRegions.values()).index('g') + 1
            an = self.hmUnknownsList[gIdx].an[Cnt]
            bn = self.hmUnknownsList[gIdx].bn[Cnt]
            an_, bn_ = np.conj(an), np.conj(bn)

            wn = 2 * nHM * pi / self.Tper
            lambdaN = self.__lambda_n(wn, urSigma)
            lambdaN_ = np.conj(lambdaN)

            aExp = cmath.exp(lambdaN * iY)
            bExp = cmath.exp(- lambdaN * iY)

            aExp_ = cmath.exp(lambdaN_ * iY)
            bExp_ = cmath.exp(- lambdaN_ * iY)

            # Fx variable declaration
            knownExpCoeffFx = an * aExp - bn * bExp
            knownExpCoeffFx_ = an_ * aExp_ + bn_ * bExp_
            termFx = lambdaN * wn * knownExpCoeffFx * knownExpCoeffFx_

            # Fy variable declaration
            coeffFy1 = lambdaN * lambdaN_
            coeffFy2 = wn ** 2
            knownExpCoeffFy1 = an * aExp - bn * bExp
            knownExpCoeffFy1_ = an_ * aExp_ - bn_ * bExp_
            termFy1 = knownExpCoeffFy1 * knownExpCoeffFy1_
            knownExpCoeffFy2 = an * aExp + bn * bExp
            knownExpCoeffFy2_ = an_ * aExp_ + bn_ * bExp_
            termFy2 = knownExpCoeffFy2 * knownExpCoeffFy2_

            # Thrust result
            Fx = termFx
            Fy = coeffFy1 * termFy1 - coeffFy2 * termFy2

            Cnt += 1

            yield Fx, Fy

    def __plotPointsAlongX(self, evenOdd, iY):

        # X axis array
        xCenterPosList = np.array([node.xCenter for node in self.matrix[iY, :]])
        dataArray = np.zeros((4, len(xCenterPosList)), dtype=np.cdouble)
        dataArray[0] = xCenterPosList

        #  Y axis array
        if evenOdd == 'even':  # even - calculated at lower node boundary in the y-direction
            yBxList = np.array([self.matrix[iY, j].BxLower for j in range(self.ppL)], dtype=np.cdouble)
            yByList = np.array([self.matrix[iY, j].ByLower for j in range(self.ppL)], dtype=np.cdouble)
            yB_List = np.array([self.matrix[iY, j].B_Lower for j in range(self.ppL)], dtype=np.cdouble)

        elif evenOdd == 'odd':  # odd - calculated at node center in the y-direction
            yBxList = np.array([self.matrix[iY, j].Bx for j in range(self.ppL)], dtype=np.cdouble)
            yByList = np.array([self.matrix[iY, j].By for j in range(self.ppL)], dtype=np.cdouble)
            yB_List = np.array([self.matrix[iY, j].B for j in range(self.ppL)], dtype=np.cdouble)

        else:
            print('neither even nor odd was chosen')
            return

        # TODO This inverts the y axis for the plot
        # bxInvertYaxis = np.flip(yBxList)
        # byInvertYaxis = np.flip(yByList)
        # dataArray[1] = bxInvertYaxis
        # dataArray[2] = byInvertYaxis

        dataArray[1] = yBxList
        dataArray[2] = yByList
        dataArray[3] = yB_List

        xSorted = np.array([i.real for i in dataArray[0]], dtype=np.float64)

        lineWidth = 2
        markerSize = 5
        tempReal = np.array([j.real for j in dataArray[1]], dtype=np.float64)
        plt.scatter(xSorted, tempReal.flatten())
        plt.plot(xSorted, tempReal.flatten(), marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('Position [m]')
        plt.ylabel('Bx [T]')
        plt.title('Bx field in airgap')
        plt.show()

        tempReal = np.array([j.real for j in dataArray[2]], dtype=np.float64)
        plt.scatter(xSorted, tempReal.flatten())
        plt.plot(xSorted, tempReal.flatten(), marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('Position [m]')
        plt.ylabel('By [T]')
        plt.title('By field in airgap')
        plt.show()

    def __buildMatAB(self):

        [reg1Count, reg2Count, reg3Count, reg4Count, reg6Count, reg7Count] = self.hmRegionsIndex
        [reg5Count] = self.mecRegionsIndex

        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        lenUnknowns = reg7Count

        print('Asize: ', self.matrixA.shape, self.matrixA.size)
        print('Bsize: ', self.matrixB.shape)
        print('(ppH, ppL, mecRegionLength)', f'({self.ppH}, {self.ppL}, {self.mecRegionLength})')

        # TODO There is potential here to use concurrent.futures.ProcessPoolExecutor since the indexing of the matrix A does not depend on previous boundary conditions
        # TODO This will require me to change the way I pass the nLoop and matBCount from boundary condition to boundary condition. Instead these need to be constants

        # -----------Boundary 1 --> Vac1 Bottom [Dirichlet]-----------
        for _ in self.n:
            self.__dirichlet(regCountOffset=reg1Count, yBoundary='-inf')

        # -----------Boundary 2 --> BackIron Bottom [HM-HM]-----------
        iY1 = self.yIndexesBackIron[0]
        iY2 = None
        bcInfo = self.__boundaryInfo(iY1, iY2, 'hmHM')
        self.__setCurrColCount(0)
        for nHM in self.n:
            # TODO We can try to cache these kind of functions for speed
            self.__hmHm(nHM, bcInfo, reg1Count, reg2Count)

        # -----------Boundary 3 --> BladeRotor Bottom [HM-HM]-----------
        iY1 = self.yIndexesBladeRotor[0]
        iY2 = None
        bcInfo = self.__boundaryInfo(iY1, iY2, 'hmHM')
        self.__setCurrColCount(0)
        for nHM in self.n:
            # TODO We can try to cache these kind of functions for speed
            self.__hmHm(nHM, bcInfo, reg2Count, reg3Count)

        # -----------Boundary 4 --> AirGap Bottom [HM-HM]-----------
        iY1 = self.yIndexesAirgap[0]
        iY2 = None
        bcInfo = self.__boundaryInfo(iY1, iY2, 'hmHM')
        self.__setCurrColCount(0)
        for nHM in self.n:
            # TODO We can try to cache these kind of functions for speed
            self.__hmHm(nHM, bcInfo, reg3Count, reg4Count)

        # -----------Boundary 5 --> MEC Bottom [MEC-HM]-----------
        iY1 = self.yIndexesMEC[0]
        iY2 = None
        bcInfo = self.__boundaryInfo(iY1, iY2, 'hmMEC')
        self.__setCurrColCount(0)
        # TODO We can try to cache these kind of functions for speed
        for nHM in self.n:
            self.__mecHm(nHM, iY1, bcInfo,
                         hmRegCountOffset=reg4Count, mecRegCountOffset=reg5Count,
                         removed_an=False, removed_bn=False, lowerUpper='lower')

        # -----------KCL EQUATIONS [MEC]-----------
        iY1 = self.yIndexesMEC[0]
        iY2 = self.yIndexesMEC[-1]
        bcInfo = self.__boundaryInfo(iY1, iY2, 'mec')

        node = 0
        i, j = iY1, 0
        while i < iY2 + 1:
            while j < self.ppL:
                # TODO We can try to cache these kind of functions for speed
                self.__mec(i, j, node, time_plex, bcInfo, reg4Count, reg6Count, reg5Count)

                node += 1
                j += 1
            j = 0
            i += 1

        self.nLoop += node
        self.matBCount += node

        # -----------Boundary 6 --> MEC Top [MEC-HM]-----------
        iY1 = self.yIndexesMEC[-1]
        iY2 = None
        bcInfo = self.__boundaryInfo(iY1, iY2, 'mecHM')
        self.__setCurrColCount(0)
        for nHM in self.n:
            # TODO We can try to cache these kind of functions for speed
            self.__mecHm(nHM, iY1, bcInfo,
                         hmRegCountOffset=reg6Count, mecRegCountOffset=reg5Count,
                         removed_an=True, removed_bn=False, lowerUpper='upper')

        # -----------Boundary 7 --> Vac2 Top-----------
        self.__setCurrColCount(0)
        for _ in self.n:
            self.__dirichlet(regCountOffset=reg6Count, yBoundary='inf')

        # Remove N equations and N coefficients at the Dirichlet boundaries that are solved in HAM_2015
        rowRemoveIdx = np.array(list(range(len(self.n))) + list(range(lenUnknowns - len(self.n), lenUnknowns)))
        # print('removeRowsIdx', rowRemoveIdx)
        colRemoveIdx = np.array(list(range(reg1Count + 1, reg2Count + 1, 2)) + list(range(reg6Count, lenUnknowns, 2)))
        # print('removeColsIdx', colRemoveIdx)

        self.__checkForErrors()

        return rowRemoveIdx, colRemoveIdx

    def finalizeCompute(self, iTol):

        print('region indexes: ', self.hmRegionsIndex, self.mecRegionsIndex, self.mecRegionLength)

        removeRows, removeCols = self.__buildMatAB()

        # matrix A and B are trimmed to remove any empty rows or columns using reduceMatrix function
        self.__reduceMatrix(removeRows, removeCols)

        # Solve for the unknown matrix X
        self.matrixX, preProcessError_matX = self.__linalg_lu(iTol)

        # update the index-tracking lists that are affected by the trimming of matrices A and B
        self.__updateLists(len(self.n))

        self.hmMatrixX = [self.matrixX[self.hmIdxs[i]] for i in range(len(self.hmIdxs))]
        self.mecMatrixX = [self.matrixX[self.mecIdxs[i]] for i in range(len(self.mecIdxs))]
        self.hmMatrixX = np.array(self.hmMatrixX, dtype=np.cdouble)
        self.mecMatrixX = np.array(self.mecMatrixX, dtype=np.cdouble)

        return preProcessError_matX

    def updateGrid(self, iErrorInX, showAirgapPlot=False):

        # Unknowns in HM regions
        matIdx = 0
        for i in self.hmRegions:
            # lower boundary
            if i == list(self.hmRegions)[0]:
                self.hmUnknownsList[i].an = self.hmMatrixX[:len(self.n)]
                # self.hmUnknownsList[i].bn = self.hmMatrixX[]
                matIdx += len(self.n)
            # upper boundary
            elif i == list(self.hmRegions)[-1]:
                # self.hmUnknownsList[i].an = self.hmMatrixX[]
                self.hmUnknownsList[i].bn = self.hmMatrixX[-len(self.n):]
                matIdx += len(self.n)
            else:
                self.hmUnknownsList[i].an = self.hmMatrixX[matIdx: matIdx + 2 * len(self.n): 2]
                self.hmUnknownsList[i].bn = self.hmMatrixX[matIdx + 1: matIdx + 2 * len(self.n): 2]
                matIdx += 2 * len(self.n)

        # Unknowns in MEC regions
        for i in range(len(self.mecRegions)):
            Cnt = 0
            i, j = self.yIndexesMEC[0], 0
            while i < self.yIndexesMEC[-1]:
                if i in self.yIndexesMEC:
                    while j < self.ppL:
                        self.matrix[i, j].Yk = self.mecMatrixX[Cnt]
                        Cnt += 1
                        j += 1
                    j = 0
                i += 1

        # Solve for B in the mesh
        i, j = 0, 0
        regCnt = 1
        while i < self.ppH:
            while j < self.ppL:

                lNode, rNode = self.neighbourNodes(j)

                if i in self.yIndexesMEC:
                    # Bottom layer of the MEC
                    if i == self.yIndexesMEC[0]:

                        ur1 = self.matrix[i - 1, 0].ur
                        sigma1 = self.matrix[i - 1, 0].sigma
                        urSigma1 = ur1 * sigma1
                        nCnt, Flux_ySum = 0, 0
                        for nHM in self.n:

                            wn = 2 * nHM * pi / self.Tper
                            lambdaN1 = self.__lambda_n(wn, urSigma1)
                            Flux_ySum += self.__postEqn21(lambdaN1, wn, self.matrix[i, j].x,
                                                          self.matrix[i, j].x + self.matrix[i, j].lx,
                                                          self.matrix[i, j].y,
                                                          self.hmUnknownsList[regCnt-1].an[nCnt],
                                                          self.hmUnknownsList[regCnt-1].bn[nCnt])
                            nCnt += 1

                        # Eqn 16
                        self.matrix[i, j].phiYp = self.__postEqn16to17(self.matrix[i + 1, j].Yk, self.matrix[i, j].Yk,
                                                                       self.matrix[i + 1, j].Ry, self.matrix[i, j].Ry)
                        # Eqn 17
                        self.matrix[i, j].phiYn = Flux_ySum

                    # Top layer of the MEC
                    elif i == self.yIndexesMEC[-1]:

                        ur2 = self.matrix[i + 1, 0].ur
                        sigma2 = self.matrix[i + 1, 0].sigma
                        urSigma2 = ur2 * sigma2
                        nCnt, Flux_ySum = 0, 0
                        for nHM in self.n:
                            wn = 2 * nHM * pi / self.Tper
                            lambdaN2 = self.__lambda_n(wn, urSigma2)
                            Flux_ySum += self.__postEqn21(lambdaN2, wn, self.matrix[i, j].x,
                                                          self.matrix[i, j].x + self.matrix[i, j].lx,
                                                          self.matrix[i + 1, j].y,
                                                          self.hmUnknownsList[regCnt + 1].an[nCnt],
                                                          self.hmUnknownsList[regCnt + 1].bn[nCnt])
                            nCnt += 1

                        # Eqn 16
                        self.matrix[i, j].phiYp = Flux_ySum
                        # Eqn 17
                        self.matrix[i, j].phiYn = self.__postEqn16to17(self.matrix[i, j].Yk, self.matrix[i - 1, j].Yk,
                                                                       self.matrix[i, j].Ry, self.matrix[i - 1, j].Ry)

                    else:
                        # Eqn 16
                        self.matrix[i, j].phiYp = self.__postEqn16to17(self.matrix[i + 1, j].Yk, self.matrix[i, j].Yk,
                                                                       self.matrix[i + 1, j].Ry, self.matrix[i, j].Ry)
                        # Eqn 17
                        self.matrix[i, j].phiYn = self.__postEqn16to17(self.matrix[i, j].Yk, self.matrix[i - 1, j].Yk,
                                                                       self.matrix[i, j].Ry, self.matrix[i - 1, j].Ry)

                    # Eqn 14
                    self.matrix[i, j].phiXp = self.__postEqn14to15(self.matrix[i, rNode].Yk, self.matrix[i, j].Yk,
                                                                   self.matrix[i, rNode].MMF, self.matrix[i, j].MMF,
                                                                   self.matrix[i, rNode].Rx, self.matrix[i, j].Rx)
                    # Eqn 15
                    self.matrix[i, j].phiXn = self.__postEqn14to15(self.matrix[i, j].Yk, self.matrix[i, lNode].Yk,
                                                                   self.matrix[i, j].MMF, self.matrix[i, lNode].MMF,
                                                                   self.matrix[i, j].Rx, self.matrix[i, lNode].Rx)

                    self.matrix[i, j].phiError = self.matrix[i, j].phiXn + self.matrix[i, j].phiYn - self.matrix[
                        i, j].phiXp - self.matrix[i, j].phiYp

                    # Eqn 40_HAM
                    self.matrix[i, j].Bx = self.postMECAvgB(self.matrix[i, j].phiXn, self.matrix[i, j].phiXp,
                                                            self.matrix[i, j].Szy)
                    self.matrix[i, j].By = self.postMECAvgB(self.matrix[i, j].phiYn, self.matrix[i, j].phiYp,
                                                            self.matrix[i, j].Sxz)

                elif i in self.yIndexesHM:

                    ur = self.matrix[i, 0].ur
                    sigma = self.matrix[i, 0].sigma
                    urSigma = ur * sigma
                    nCnt, BxSumCenter, BySumCenter, BxSumLower, BySumLower = 0, 0, 0, 0, 0
                    for nHM in self.n:
                        wn = 2 * nHM * pi / self.Tper
                        lambdaN = self.__lambda_n(wn, urSigma)
                        an = self.hmUnknownsList[regCnt].an[nCnt]
                        bn = self.hmUnknownsList[regCnt].bn[nCnt]
                        BxSumCenter += self.__postEqn8(lambdaN, wn,
                                                       self.matrix[i, j].xCenter, self.matrix[i, j].yCenter, an, bn)
                        BxSumLower += self.__postEqn8(lambdaN, wn,
                                                      self.matrix[i, j].xCenter, self.matrix[i, j].y, an, bn)
                        BySumCenter += self.__postEqn9(lambdaN, wn,
                                                       self.matrix[i, j].xCenter, self.matrix[i, j].yCenter, an, bn)
                        BySumLower += self.__postEqn9(lambdaN, wn,
                                                      self.matrix[i, j].xCenter, self.matrix[i, j].y, an, bn)

                        nCnt += 1

                    self.matrix[i, j].Bx = BxSumCenter
                    self.matrix[i, j].BxLower = BxSumLower

                    self.matrix[i, j].By = BySumCenter
                    self.matrix[i, j].ByLower = BySumLower

                else:
                    print('we cant be here')
                    return

                self.matrix[i, j].B = cmath.sqrt(self.matrix[i, j].Bx ** 2 + self.matrix[i, j].By ** 2)

                # Counter for each HM region
                if i in [self.yIndexesVacLower[-1], self.yIndexesBackIron[-1], self.yIndexesBladeRotor[-1],
                         self.yIndexesAirgap[-1], self.yIndexesMEC[-1], self.yIndexesVacUpper[-1]] and j == self.ppL - 1:
                    regCnt += 1

                j += 1
            j = 0
            i += 1

        # This is a nested generator comprehension for phiError
        genPhiError = (j.phiError for i in self.matrix for j in i)
        postProcessError_Phi = max(genPhiError)
        # TODO This is inefficient and temp for testing
        genPhiError_min = (j.phiError for i in self.matrix for j in i)
        print(
            f'max error post processing is: {postProcessError_Phi} vs min error post processing: {min(genPhiError_min)} vs error in matX: {iErrorInX}')

        allowableError = 10 ** (-14)
        if postProcessError_Phi > allowableError or iErrorInX > allowableError:
            self.writeErrorToDict(key='name',
                                  error=Error.buildFromScratch(name='violatedKCL',
                                                               description="ERROR - Kirchhoff's current law is violated",
                                                               cause=True))

        # Thrust Calculation
        centerAirgapIdx_y = self.yIndexesAirgap[0] + self.ppAirgap // 2
        if self.ppAirgap % 2 == 0:  # even
            evenOdd = 'even'
            centerAirgap_y = self.matrix[centerAirgapIdx_y][0].y
        else:  # odd
            evenOdd = 'odd'
            centerAirgap_y = self.matrix[centerAirgapIdx_y][0].yCenter

        ur = self.matrix[self.yIndexesAirgap[0], 0].ur
        sigma = self.matrix[self.yIndexesAirgap[0], 0].sigma

        resFx, resFy = np.cdouble(0), np.cdouble(0)
        thrustGenerator = self.__genForces(ur * sigma, centerAirgap_y)
        for (x, y) in thrustGenerator:
            resFx += x
            resFy += y

        resFx *= - j_plex * self.D * self.Tper / (2 * uo)
        resFy *= - self.D * self.Tper / (4 * uo)
        print(f'Fx: {round(resFx.real, 2)}N,', f'Fy: {round(resFy.real, 2)}N')

        if showAirgapPlot:
            self.__plotPointsAlongX(evenOdd, centerAirgapIdx_y)


# noinspection PyGlobalUndefined
def complexFourierTransform(model_in, harmonics_in):
    """
    This function was written to plot the Bx field at the boundary between the coils and the airgap,
     described in equation 24 of the 2019 paper. The Bx field is piecewise-continuous and is plotted in Blue.
     The complex Fourier transform was applied to the Bx field and plotted in Red. The accuracy of the
     complex Fourier transform depends on: # of harmonics, # of x positions, # of nodes in the x-direction of the model

    A perfect complex Fourier transform extends harmonics to +-Inf, while the 0th harmonic is accounted for in the
     c_0 term. Since the Bx field does not have a y-direction offset, this term can be neglected.
    """

    global row_upper_FT, expandedLeftNodeEdges, slices, idx_FT, harmonics, model

    model = model_in

    idx_FT = 0
    harmonics = harmonics_in

    yIdx_lower = model.yIndexesMEC[0]
    yIdx_upper = model.yIndexesMEC[-1]
    row_upper_FT = model.matrix[yIdx_upper]
    leftNodeEdges = [node.x for node in row_upper_FT]
    slices = 10
    outerIdx, increment = 0, 0
    expandedLeftNodeEdges = list(np.zeros(len(leftNodeEdges) * slices, dtype=np.float64))
    for idx in range(len(expandedLeftNodeEdges)):
        if idx % slices == 0:
            increment = 0
            if idx != 0:
                outerIdx += 1
        else:
            increment += 1
        sliceWidth = row_upper_FT[outerIdx].lx / slices
        expandedLeftNodeEdges[idx] = row_upper_FT[outerIdx].x + increment * sliceWidth

    # noinspection PyGlobalUndefined
    def fluxAtBoundary():
        global row_upper_FT, idx_FT, model

        lNode, rNode = model.neighbourNodes(idx_FT)
        phiXn = (row_upper_FT[idx_FT].MMF + row_upper_FT[lNode].MMF) \
                / (row_upper_FT[idx_FT].Rx + row_upper_FT[lNode].Rx)
        phiXp = (row_upper_FT[idx_FT].MMF + row_upper_FT[rNode].MMF) \
                / (row_upper_FT[idx_FT].Rx + row_upper_FT[rNode].Rx)
        return phiXn, phiXp

    # noinspection PyGlobalUndefined
    @lru_cache(maxsize=5)
    def pieceWise_upper(x_in):
        global row_upper_FT, idx_FT, model

        if x_in in leftNodeEdges[1:]:
            idx_FT += 1

        phiXn, phiXp = fluxAtBoundary()

        return model.postMECAvgB(phiXn, phiXp, row_upper_FT[idx_FT].Szy)

    # noinspection PyGlobalUndefined
    @lru_cache(maxsize=5)
    def fourierSeries(x_in):
        global row_upper_FT, idx_FT, model
        sumN = 0.0
        for nHM in harmonics:
            wn = 2 * nHM * pi / model.Tper
            coeff = j_plex / (wn * model.Tper)
            sumK = 0.0
            for iX in range(len(row_upper_FT)):
                idx_FT = iX
                phiXn, phiXp = fluxAtBoundary()
                f = (phiXn + phiXp) / (2 * row_upper_FT[iX].Szy)
                Xl = row_upper_FT[iX].x
                Xr = Xl + row_upper_FT[iX].lx
                resExp = cmath.exp(-j_plex * wn * (Xr - model.vel * model.t))\
                         - cmath.exp(-j_plex * wn * (Xl - model.vel * model.t))

                sumK += f * resExp

            sumN += coeff * sumK * cmath.exp(j_plex * wn * x_in)

        return sumN

    vfun = np.vectorize(pieceWise_upper)
    idx_FT = 0
    x = expandedLeftNodeEdges
    y1 = vfun(x)
    # vfun = np.vectorize(fourierSeries)
    # y2 = vfun(x)
    #
    # plt.plot(x, y1, 'b-')
    # plt.plot(x, y2, 'r-')
    # plt.xlabel('Position [m]')
    # plt.ylabel('Bx [T]')
    # plt.title('Bx field at airgap Boundary')
    # plt.show()

    return x, y1


def plotFourierError():

    iterations = 1
    step = 2
    start = 10
    pixDivs = range(start, start + iterations * step, step)
    modelList = np.empty(len(pixDivs), dtype=ndarray)

    lowDiscrete = 30
    n = range(-lowDiscrete, lowDiscrete + 1)
    n = np.delete(n, len(n) // 2, 0)
    slots = 16
    poles = 6
    wt, ws = 6 / 1000, 10 / 1000
    slotpitch = wt + ws
    endTeeth = 2 * (4 / 3 * wt)
    length = ((slots - 1) * slotpitch + ws) + endTeeth
    meshDensity = np.array([4, 2])
    xMeshIndexes = [[0, 0]] + [[0, 0]] + [[0, 0], [0, 0]] * (slots - 1) + [[0, 0]] + [[0, 0]] + [[0, 0]]
    # [LowerVac], [Yoke], [LowerSlots], [UpperSlots], [Airgap], [BladeRotor], [BackIron], [UpperVac]
    yMeshIndexes = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    canvasSpacing = 80

    for idx, pixelDivisions in enumerate(pixDivs):

        pixelSpacing = slotpitch / pixelDivisions
        loopedModel = Model.buildFromScratch(slots=slots, poles=poles, length=length, n=n, pixelSpacing=pixelSpacing,
                                             canvasSpacing=canvasSpacing,
                                             meshDensity=meshDensity, meshIndexes=[xMeshIndexes, yMeshIndexes],
                                             hmRegions=np.array([0, 2, 3, 4, 5], dtype=np.int16),
                                             mecRegions=np.array([1], dtype=np.int16))
        loopedModel.buildGrid(pixelSpacing=pixelSpacing, meshIndexes=[xMeshIndexes, yMeshIndexes])
        loopedModel.finalizeGrid(pixelDivisions)
        modelList[idx] = ((pixelDivisions, loopedModel.ppL, loopedModel.ppH), complexFourierTransform(loopedModel, n))

    for idx, ((pixelDivisions, ppL, ppH), (xSequence, ySequence)) in enumerate(modelList):
        plt.plot(xSequence, ySequence, label=f'PixelDivs: {pixelDivisions}, (ppL, ppH): {(ppL, ppH)}')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    # profile_main()  # To profile the main execution
    plotFourierError()