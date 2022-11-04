from LIM.Grid import *
from LIM.SlotPoleCalculation import np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from functools import lru_cache


# This performs the lower-upper decomposition of A to solve for x in Ax = B
# A must be a square matrix
# @njit("float64[:](float64[:, :], float64[:])", cache=True)
class Model(Grid):

    currColCount, currColCountUpper, rowCount = 0, 0, 0

    def __init__(self, kwargs, buildFromJson=False, buildBaseline=False):

        if buildFromJson:
            for attr_key in kwargs:
                self.__dict__[attr_key] = kwargs[attr_key]
            return

        super().__init__(kwargs, buildBaseline)

        self.errorTolerance = kwargs['hamCfg']['errorTolerance']

        self.hmUnknownsList = {}
        self.canvasRowRegIdxs, self.canvasColRegIdxs = [], []
        self.mecIdxs = []
        self.hmMatrixX, self.mecMatrixX = [], []
        self.vacLowerRow, self.vacUpperRow = [], []

        self.lenUnknowns = int(self.hmRegionsIndex[-1] if not self.allMecRegions else self.mecRegionsIndex[-1])

        self.matrixA = np.zeros((self.lenUnknowns, self.lenUnknowns), dtype=np.cdouble)
        self.matrixB = np.zeros(len(self.matrixA), dtype=np.cdouble)
        self.matrixX = np.zeros(len(self.matrixA), dtype=np.cdouble)

        for cnt, i in enumerate(self.hmRegions):
            if cnt == 0:
                self.hmUnknownsList[i] = Region.buildFromScratch(type=self.hmRegions[i], an=np.zeros(len(self.n)), bn=None)
            elif cnt == len(self.hmRegions) - 1:
                self.hmUnknownsList[i] = Region.buildFromScratch(type=self.hmRegions[i], an=None, bn=np.zeros(len(self.n)))
            else:
                self.hmUnknownsList[i] = Region.buildFromScratch(type=self.hmRegions[i], an=np.zeros(len(self.n)), bn=np.zeros(len(self.n)))

        # HM and MEC unknown indexes in matrix A and B, used for visualization
        indexRowVal, indexColVal = 0, 0
        for region in self.getFullRegionDict():
            if region != 'mec' and region.split('_')[0] != 'vac':
                indexColVal += 2 * len(self.n)
                indexRowVal += 2 * len(self.n)
                self.canvasRowRegIdxs.append(indexRowVal)

            elif region.split('_')[0] == 'vac':
                indexColVal += len(self.n)

            elif region == 'mec':
                indexColVal += self.mecRegionLength
                for bc in self.getFullRegionDict()[region]['bc'].split(', '):
                    if bc == 'mecHm':
                        indexRowVal += len(self.n)
                    elif bc == 'mec' and not self.allMecRegions:
                        indexRowVal += self.mecRegionLength
                    else:
                        indexRowVal, indexColVal = 0, 0

                    self.canvasRowRegIdxs.append(indexRowVal)

            self.canvasColRegIdxs.append(indexColVal)
        self.mecCanvasRegIdxs = [self.canvasRowRegIdxs[list(self.getFullRegionDict()).index('mec')-1] + self.ppL * i for i in range(1, self.ppMEC)]

        for i in self.mecRegionsIndex if not self.allMecRegions else self.mecRegionsIndex[:-1]:
            self.mecIdxs.extend(list(range(i, i + self.mecRegionLength)))
        self.hmIdxs = [i for i in range(len(self.matrixX)) if i not in self.mecIdxs]

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    @classmethod
    def buildBaseline(cls, **kwargs):
        return cls(kwargs=kwargs, buildBaseline=True)

    def equals(self, otherObject, removedAtts):
        equality = True
        if not isinstance(otherObject, Model):
            # don't attempt to compare against unrelated types
            equality = False

        else:
            # Compare items between objects
            selfItems = list(filter(lambda x: True if x[0] not in removedAtts else False, self.__dict__.items()))
            otherItems = list(otherObject.__dict__.items())
            for cnt, (entryKey, entryVal) in enumerate(selfItems):
                otherKey = otherItems[cnt][0]
                otherVal = otherItems[cnt][1]
                # Check keys are equal
                if entryKey == otherKey:
                    # np.ndarray were converted to lists to dump data to json object, now compare against those lists
                    if type(entryVal) == np.ndarray:
                        # Greater than 1D array
                        if len(entryVal.shape) > 1:
                            # Check that 2D array are equal
                            if np.all(np.all(entryVal != otherVal, axis=1)):
                                equality = False
                        # 1D array
                        else:
                            if entryVal.tolist() != otherVal:
                                # The contents of the array or list may be a deconstructed complex type
                                if np.all(list(map(lambda val: val[1] == rebuildPlex(otherVal[val[0]]), enumerate(entryVal)))):
                                    pass
                                else:
                                    equality = False
                    else:
                        if entryVal != otherVal:
                            # Check to see if keys of the value are equal to string versions of otherItems keys
                            try:
                                if list(map(lambda x: str(x), entryVal.keys())) != list(otherVal.keys()):
                                    equality = False
                            except AttributeError:
                                pass
                else:
                    equality = False

        return equality

    def hmHm(self, nHM, listBCInfo, RegCountOffset1, RegCountOffset2, remove_an, remove_bn):

        hb1, ur1, urSigma1, _, ur2, urSigma2 = listBCInfo

        wn = 2 * nHM * pi / self.Tper
        lambdaN1 = self.__lambda_n(wn, urSigma1)
        lambdaN2 = self.__lambda_n(wn, urSigma2)

        # By Condition
        Ba_lower, Ba_upper, Bb_lower, Bb_upper = self.__preEqn24_2018(lambdaN1, lambdaN2, hb1)
        self.matrixA[self.rowCount, RegCountOffset1 + self.currColCount] = Ba_lower  # an Lower
        if not remove_bn:
            self.matrixA[self.rowCount, RegCountOffset1 + 1 + self.currColCount] = Bb_lower  # bn Lower
        if not remove_an:
            self.matrixA[self.rowCount, RegCountOffset2 + self.currColCountUpper] = Ba_upper  # an Upper
            self.matrixA[self.rowCount, RegCountOffset2 + 1 + self.currColCountUpper] = Bb_upper  # bn Upper
        else:
            self.matrixA[self.rowCount, RegCountOffset2 + self.currColCountUpper] = Bb_upper  # bn Upper

        # Hx Condition
        Ha_lower, Ha_upper, Hb_lower, Hb_upper = self.__preEqn25_2018(lambdaN1, lambdaN2, ur1, ur2, hb1)

        self.matrixA[self.rowCount + 1, RegCountOffset1 + self.currColCount] = Ha_lower  # an Lower
        if not remove_bn:
            self.matrixA[self.rowCount + 1, RegCountOffset1 + 1 + self.currColCount] = Hb_lower  # bn Lower
        if not remove_an:
            self.matrixA[self.rowCount + 1, RegCountOffset2 + self.currColCountUpper] = Ha_upper  # an Upper
            self.matrixA[self.rowCount + 1, RegCountOffset2 + 1 + self.currColCountUpper] = Hb_upper  # bn Upper
        else:
            self.matrixA[self.rowCount + 1, RegCountOffset2 + self.currColCountUpper] = Hb_upper  # bn Upper

        self.incrementCurrColCnt(remove_an, remove_bn)
        self.rowCount += 2

    def mecHm(self, nHM, iY, listBCInfo, hmRegCountOffset, mecRegCountOffset, removed_an, removed_bn, lowerUpper):
        hb, ur, urSigma = listBCInfo
        wn = 2 * nHM * pi / self.Tper
        lambdaN = self.__lambda_n(wn, urSigma)
        time_plex = self.timePlex()
        coeff = 2 * j_plex / (wn * self.Tper)

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='iYnotInMEC',
                                                           description='ERROR - The index iY is not in the MEC boundaries',
                                                           cause=iY not in [self.yIndexesMEC[0], self.yIndexesMEC[-1]]))

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
                self.matrixA[self.rowCount, mecRegCountOffset + iX] = combinedPsi_k  # Curr
            elif lowerUpper == 'upper':
                self.matrixA[self.rowCount, mecRegCountOffset + self.mecRegionLength - self.ppL + iX] = combinedPsi_k  # Curr

            sumResEqn22Source += (resSource_kn + resSource_k + resSource_kp)

        # HM related equations
        hmResA, hmResB = self.__preEqn8(ur, lambdaN, wn, hb)

        # If the neighbouring region is a Dirichlet region, an or bn may be removed
        # a not b
        if not removed_an and removed_bn:
            self.matrixA[self.rowCount, hmRegCountOffset + self.currColCount] = hmResA  # an
        # b not a
        elif removed_an and not removed_bn:
            self.matrixA[self.rowCount, hmRegCountOffset + self.currColCountUpper] = hmResB  # bn
        # a and b
        elif not removed_an and not removed_bn:
            self.matrixA[self.rowCount, hmRegCountOffset + self.currColCount] = hmResA  # an
            self.matrixA[self.rowCount, hmRegCountOffset + 1 + self.currColCount] = hmResB  # bn
        # a and b
        else:
            print('There was an error here')
            return None

        self.matrixB[self.rowCount] = - coeff * sumResEqn22Source

        self.incrementCurrColCnt(removed_an, removed_bn)
        self.rowCount += 1

    def mec(self, i, j, node, time_plex, listBCInfo, mecRegCountOffset,
            hmRegCountOffset1=None, hmRegCountOffset2=None, removed_an=None, removed_bn=None):

        hb1, urSigma1, hb2, urSigma2 = listBCInfo
        lNode, rNode = self.neighbourNodes(j)
        if i == 0:
            _, vacResY = self.matrix[i, j].getReluctance(self, isVac=True)
            northRelDenom = self.matrix[i, j].Ry + self.matrix[i + 1, j].Ry
            southRelDenom = self.matrix[i, j].Ry + vacResY
        elif i == self.ppH - 1:
            _, vacResY = self.matrix[i, j].getReluctance(self, isVac=True)
            northRelDenom = self.matrix[i, j].Ry + vacResY
            southRelDenom = self.matrix[i, j].Ry + self.matrix[i - 1, j].Ry
        else:
            northRelDenom = self.matrix[i, j].Ry + self.matrix[i + 1, j].Ry
            southRelDenom = self.matrix[i, j].Ry + self.matrix[i - 1, j].Ry
        eastRelDenom = self.matrix[i, j].Rx + self.matrix[i, rNode].Rx
        westRelDenom = self.matrix[i, j].Rx + self.matrix[i, lNode].Rx
        eastMMFNum = self.matrix[i, j].MMF + self.matrix[i, rNode].MMF
        westMMFNum = self.matrix[i, j].MMF + self.matrix[i, lNode].MMF

        currIdx = mecRegCountOffset + node
        northIdx = currIdx + self.ppL
        southIdx = currIdx - self.ppL

        # Bottom layer of the mesh
        if i == self.yIndexesMEC[0]:
            if not self.allMecRegions:
                self.__setCurrColCount(0, 0)
                # South Node
                for nHM in self.n:
                    wn = 2 * nHM * pi / self.Tper
                    lambdaN1 = self.__lambda_n(wn, urSigma1)
                    anCoeff, bnCoeff = self.__preEqn21(lambdaN1, wn, self.matrix[i, j].x,
                                                       self.matrix[i, j].x + self.matrix[i, j].lx, hb1)

                    self.matrixA[currIdx, hmRegCountOffset1 + self.currColCount] = anCoeff  # an
                    if not removed_bn:
                        self.matrixA[currIdx, hmRegCountOffset1 + 1 + self.currColCount] = bnCoeff  # bn

                    self.incrementCurrColCnt(False, removed_bn)

                # North Node
                self.matrixA[currIdx, northIdx] = - time_plex / northRelDenom

                # Current Node
                self.matrixA[currIdx, currIdx] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / northRelDenom)

            else:
                # North Node
                self.matrixA[currIdx, northIdx] = - time_plex / northRelDenom
                # Current Node
                self.matrixA[currIdx, currIdx] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / northRelDenom + 1 / southRelDenom)

        # Top layer of the mesh
        elif i == self.yIndexesMEC[-1]:
            if not self.allMecRegions:
                self.__setCurrColCount(0, 0)
                # North Node
                for nHM in self.n:
                    wn = 2 * nHM * pi / self.Tper
                    lambdaN2 = self.__lambda_n(wn, urSigma2)
                    anCoeff, bnCoeff = self.__preEqn21(lambdaN2, wn, self.matrix[i, j].x,
                                                       self.matrix[i, j].x + self.matrix[i, j].lx, hb2)

                    if removed_an:
                        self.matrixA[currIdx, hmRegCountOffset2 + self.currColCountUpper] = - bnCoeff  # bn
                    else:
                        self.matrixA[currIdx, hmRegCountOffset2 + self.currColCountUpper] = - anCoeff  # an
                        self.matrixA[currIdx, hmRegCountOffset2 + 1 + self.currColCountUpper] = - bnCoeff  # bn

                    self.incrementCurrColCnt(removed_an, False)

                # South Node
                self.matrixA[currIdx, southIdx] = - time_plex / southRelDenom

                # Current Node
                self.matrixA[currIdx, currIdx] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / southRelDenom)

            else:
                # South Node
                self.matrixA[currIdx, southIdx] = - time_plex / southRelDenom
                # Current Node
                self.matrixA[currIdx, currIdx] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / northRelDenom + 1 / southRelDenom)

        else:
            # North Node
            self.matrixA[currIdx, northIdx] = - time_plex / northRelDenom
            # South Node
            self.matrixA[currIdx, southIdx] = - time_plex / southRelDenom
            # Current Node
            self.matrixA[currIdx, currIdx] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / northRelDenom + 1 / southRelDenom)

        # West Edge
        if node % self.ppL == 0:
            westIdx = northIdx - 1
            eastIdx = currIdx + 1
        # East Edge
        elif (node + 1) % self.ppL == 0:
            westIdx = currIdx - 1
            eastIdx = southIdx + 1
        else:
            westIdx = currIdx - 1
            eastIdx = currIdx + 1

        # West Node
        self.matrixA[currIdx, westIdx] = - time_plex / westRelDenom
        # East Node
        self.matrixA[currIdx, eastIdx] = - time_plex / eastRelDenom

        # Result
        self.matrixB[currIdx] = eastMMFNum / eastRelDenom - westMMFNum / westRelDenom

    @staticmethod
    def __preEqn24_2018(lam_lower, lam_upper, y):

        resA_lower = cmath.exp(lam_lower * y)
        resA_upper = - cmath.exp(lam_upper * y)
        resB_lower = cmath.exp(-lam_lower * y)
        resB_upper = - cmath.exp(-lam_upper * y)

        return resA_lower, resA_upper, resB_lower, resB_upper

    @staticmethod
    def __preEqn25_2018(lam_lower, lam_upper, ur_lower, ur_upper, y):

        resA_lower = (lam_lower / ur_lower) * cmath.exp(lam_lower * y)
        resA_upper = - (lam_upper / ur_upper) * cmath.exp(lam_upper * y)
        resB_lower = - (lam_lower / ur_lower) * cmath.exp(-lam_lower * y)
        resB_upper = (lam_upper / ur_upper) * cmath.exp(-lam_upper * y)

        return resA_lower, resA_upper, resB_lower, resB_upper

    @staticmethod
    def postMECAvgB(fluxN, fluxP, S_xyz):

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

    def __postEqn14to15(self, destpot, startpot, destMMF, startMMF, destRel, startRel):
        return (self.timePlex() * (destpot - startpot) + destMMF + startMMF) / (destRel + startRel)

    def __postEqn16to17(self, destpot, startpot, destRel, startRel):
        return self.timePlex() * (destpot - startpot) / (destRel + startRel)

    def __preEqn21(self, lam, wn, Xl, Xr, y):

        res1 = cmath.exp(j_plex * wn * Xl) - cmath.exp(j_plex * wn * Xr)
        res2 = self.timePlex()

        aExp = cmath.exp(lam * y)
        bExp = cmath.exp(-lam * y)

        resA = self.D * res1 * res2 * aExp
        resB = self.D * res1 * res2 * bExp

        return resA, resB

    def __postEqn21(self, lam, wn, Xl, Xr, y, an, bn):

        res1 = an * cmath.exp(lam * y) + bn * cmath.exp(-lam * y)
        res2 = cmath.exp(j_plex * wn * Xl) - cmath.exp(j_plex * wn * Xr)
        res3 = self.timePlex()
        res = self.D * res1 * res2 * res3

        return res

    def __eqn23Integral(self, wn, Xl, Xr, ur, Szy):

        coeff = 1 / (2 * ur * Szy)
        res = coeff * (cmath.exp(-j_plex * wn * (Xr - self.vel * self.t)) -
                       cmath.exp(-j_plex * wn * (Xl - self.vel * self.t)))

        return res

    # ___Manipulate Matrix Methods___ #
    def __checkForErrors(self):

        areDimsSame = all(val == self.rowCount for val in [self.matrixA.shape[0], self.matrixB.size])
        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='Dims Mismatch',
                                                           description='Error - The matrices A and B do not match',
                                                           cause=not areDimsSame))

        if not self.allMecRegions:
            self.writeErrorToDict(key='name',
                                  error=Error.buildFromScratch(name='regCount',
                                                               description='Error - regCount',
                                                               cause=self.hmRegionsIndex[-1] != self.matrixA.shape[1]))

        # Check rows
        zeroRowsList = np.all(self.matrixA == 0, axis=1)
        removeRowIdx = np.where(zeroRowsList)
        print('Zero Row Indexes: ', removeRowIdx[0])

        # Check Columns
        zeroColsList = np.all(self.matrixA == 0, axis=0)
        removeColIdx = np.where(zeroColsList)
        print('Zero Column Indexes: ', removeColIdx[0])

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
            ur1, sigma1 = self.__getLowerUrSigma(iY1)
            urSigma1 = ur1 * sigma1

            hb2 = self.matrix[iY2+1, 0].y if iY2 != self.ppH - 1 else self.modelHeight
            ur2, sigma2 = self.__getUpperUrSigma(iY2)
            urSigma2 = ur2 * sigma2

            return [hb1, urSigma1, hb2, urSigma2]

        elif boundaryType == 'hmHm' and not iY2:
            # Case Upper Dirichlet
            if iY1 == self.ppH:
                hb1 = self.modelHeight
                ur2 = self.air.ur
                sigma2 = self.air.sigma
            else:
                hb1 = self.matrix[iY1, 0].y
                ur2 = self.matrix[iY1, 0].ur
                sigma2 = self.matrix[iY1, 0].sigma
            ur1, sigma1 = self.__getLowerUrSigma(iY1)
            urSigma1 = ur1 * sigma1
            urSigma2 = ur2 * sigma2

            return [hb1, ur1, urSigma1, None, ur2, urSigma2]

        elif boundaryType == 'hmMec' and not iY2:
            hb = self.matrix[iY1, 0].y
            ur, sigma = self.__getLowerUrSigma(iY1)
            urSigma = ur * sigma

            return [hb, ur, urSigma]

        elif boundaryType == 'mecHm' and not iY2:
            hb = self.matrix[iY1 + 1, 0].y if iY1 != self.ppH - 1 else self.modelHeight
            ur, sigma = self.__getUpperUrSigma(iY1)
            urSigma = ur * sigma

            return [hb, ur, urSigma]

        else:
            print('An incorrect boundary type was chosen')

    def __linalg_lu(self):

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='linalg deepcopy',
                                                           description="ERROR - Matrix A and B should be ndarray so matrix deep copy is not performed",
                                                           cause=type(self.matrixA) != np.ndarray or type(self.matrixB) != np.ndarray))

        if self.matrixA.shape[0] == self.matrixA.shape[1]:
            lu, piv = lu_factor(self.matrixA)
            resX = lu_solve((lu, piv), self.matrixB)
            remainder = self.matrixA @ resX - self.matrixB
            print(f'This was the max error seen in the solution for x: {max(remainder)} vs min: {min(remainder)}')
            testPass = np.allclose(remainder, np.zeros((len(self.matrixA),)), atol=self.errorTolerance)
            print(f'LU Decomp test: {testPass}, if True then x is a solution of Ax = iB with a tolerance of {self.errorTolerance}')
            if testPass:
                return resX, max(remainder)
            else:
                print('LU Decomp test failed')
                return
        else:
            print('Error - A is not a square matrix')
            return

    def __setCurrColCount(self, v1, v2):
        self.currColCount = v1
        self.currColCountUpper = v2

    def incrementCurrColCnt(self, removed_an, removed_bn):
        if removed_an:
            self.currColCount += 2
            self.currColCountUpper += 1
        elif removed_bn:
            self.currColCount += 1
            self.currColCountUpper += 2
        else:
            self.currColCount += 2
            self.currColCountUpper += 2

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='RemoveAnBn',
                                                           description='ERROR - The remove an and bn parameters cannot both be true',
                                                           cause=removed_an and removed_bn))

    def __lambda_n(self, wn, urSigma):
        return cmath.sqrt(wn ** 2 + j_plex * uo * urSigma * (2 * pi * self.f + wn * self.vel))

    def __genForces(self, urSigma, iY):

        Cnt = 0
        for nHM in self.n:
            gIdx = list(self.hmRegions.keys())[list(self.hmRegions.values()).index('g')]
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

    def __plotPointsAlongHM(self, iY):
        lineWidth = 2
        markerSize = 5
        plt.scatter(self.n, self.hmUnknownsList[iY].an)
        plt.plot(self.n, self.hmUnknownsList[iY].an, marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.scatter(self.n, self.hmUnknownsList[iY].bn)
        plt.plot(self.n, self.hmUnknownsList[iY].bn, marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('harmonic')
        plt.ylabel('magnitude')
        plt.title('Solved Unknown Airgap')
        plt.show()

        plt.scatter(self.n, self.hmUnknownsList[iY-1].an)
        plt.plot(self.n, self.hmUnknownsList[iY-1].an, marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.scatter(self.n, self.hmUnknownsList[iY-1].bn)
        plt.plot(self.n, self.hmUnknownsList[iY-1].bn, marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('harmonic')
        plt.ylabel('magnitude')
        plt.title('Solved Unknown Airgap')
        plt.show()

    def __plotPointsAlongX(self, iY, invertY=False):

        lineWidth = 2
        markerSize = 5

        # X axis array
        xCenterPosList = np.array([node.xCenter for node in self.matrix[iY, :]])
        dataArray = np.zeros((4, len(xCenterPosList)), dtype=np.cdouble)
        dataArray[0] = xCenterPosList
        xSorted = np.array([i.real for i in dataArray[0]], dtype=np.float64)

        #  Y axis array
        yBxList = np.array([self.matrix[iY, j].Bx for j in range(self.ppL)], dtype=np.cdouble)
        yByList = np.array([self.matrix[iY, j].By for j in range(self.ppL)], dtype=np.cdouble)

        # This flips the plot about the x-axis
        if invertY:
            dataArray[1] = np.array(list(map(lambda x: -1 * x, yBxList)))
            dataArray[2] = np.array(list(map(lambda x: -1 * x, yByList)))
        else:
            dataArray[1] = yBxList
            dataArray[2] = yByList

        tempReal = np.array([j.real for j in dataArray[1]], dtype=np.float64)

        # Background shading slots
        minY1, maxY1 = min(tempReal), max(tempReal)
        xPosLines = list(map(lambda x: x.x, self.matrix[0][self.slotArray[::self.ppSlot]]))
        for cnt, each in enumerate(xPosLines):
            plt.axvspan(each, each + self.ws, facecolor='b', alpha=0.15, label="_"*cnt + "slot regions")
        plt.legend()

        plt.scatter(xSorted, tempReal.flatten())
        plt.plot(xSorted, tempReal.flatten(), marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('Position [m]')
        plt.ylabel('Bx [T]')
        plt.title('Bx field in airgap')
        plt.show()

        tempReal = np.array([j.real for j in dataArray[2]], dtype=np.float64)

        # Background shading slots
        minY1, maxY1 = min(tempReal), max(tempReal)
        xPosLines = list(map(lambda x: x.x, self.matrix[0][self.slotArray[::self.ppSlot]]))
        for cnt, each in enumerate(xPosLines):
            plt.axvspan(each, each + self.ws, facecolor='b', alpha=0.15, label="_"*cnt + "slot regions")
        plt.legend()

        plt.scatter(xSorted, tempReal.flatten())
        plt.plot(xSorted, tempReal.flatten(), marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('Position [m]')
        plt.ylabel('By [T]')
        plt.title('By field in airgap')
        plt.show()

    def __getYboundaryIncludeMEC(self):
        yBoundaryIncludeMec = [index for index in self.yBoundaryList if index not in self.yIndexesMEC]
        yBoundaryIncludeMec.extend([self.yIndexesMEC[0], self.yIndexesMEC[-1]])
        yBoundaryIncludeMec.sort()
        return yBoundaryIncludeMec

    def __getLowerUrSigma(self, i):
        if i == 0:
            ur = self.air.ur
            sigma = self.air.sigma
        else:
            ur = self.matrix[i - 1, 0].ur
            sigma = self.matrix[i - 1, 0].sigma
        return ur, sigma

    def __getUpperUrSigma(self, i):
        if i == self.ppH - 1:
            ur = self.air.ur
            sigma = self.air.sigma
        else:
            ur = self.matrix[i + 1, 0].ur
            sigma = self.matrix[i + 1, 0].sigma
        return ur, sigma

    def __buildMatAB(self):

        yBoundaryIncludeMec = self.__getYboundaryIncludeMEC()

        hmMec = True
        cnt, hmCnt, mecCnt = 0, 0, 0
        iY1, nextY1 = 0, yBoundaryIncludeMec[0] + 1
        for region in self.getFullRegionDict():
            if region.split('_')[0] != 'vac':
                prevReg, nextReg = self.getLastAndNextRegionName(region)
                if nextReg is not None:
                    if nextReg.split('_')[0] != 'vac':
                        _, nextnextReg = self.getLastAndNextRegionName(nextReg)
                for bc in self.getFullRegionDict()[region]['bc'].split(', '):

                    # Mec region calculation
                    if bc == 'mec':
                        node = 0
                        iY1, iY2 = self.yIndexesMEC[0], self.yIndexesMEC[-1]
                        i, j = iY1, 0
                        while i < iY2 + 1:
                            while j < self.ppL:
                                params = {'i': i, 'j': j, 'node': node, 'time_plex': self.timePlex(),
                                          'listBCInfo': self.__boundaryInfo(iY1, iY2, bc),
                                          'mecRegCountOffset': self.mecRegionsIndex[mecCnt]}
                                if not self.allMecRegions:
                                    params['hmRegCountOffset1'] = self.hmRegionsIndex[hmCnt]
                                    params['removed_bn'] = True if prevReg.split('_')[0] == 'vac' else False
                                    params['hmRegCountOffset2'] = self.hmRegionsIndex[hmCnt + 1]
                                    params['removed_an'] = True if nextReg.split('_')[0] == 'vac' else False

                                # TODO We can try to cache these kind of functions for speed
                                getattr(self, bc)(**params)

                                node += 1
                                j += 1
                            j = 0
                            i += 1
                        # Increment the indexing after finishing with the mec region
                        self.__setCurrColCount(0, 0)
                        self.rowCount += node
                        nextY1 = iY2 + 1
                        cnt += 1
                        hmCnt += 1

                    # All boundary conditions loop through N harmonics except mec
                    else:
                        # Loop through harmonics and calculate each boundary condition
                        if bc == 'hmHm':
                            params = {'listBCInfo': self.__boundaryInfo(iY1, None, bc),
                                      'RegCountOffset1': self.hmRegionsIndex[hmCnt],
                                      'RegCountOffset2': self.hmRegionsIndex[hmCnt+1],
                                      'remove_an': True if nextReg.split('_')[0] == 'vac' else False,
                                      'remove_bn': True if prevReg.split('_')[0] == 'vac' else False}

                        elif bc == 'mecHm':
                            iY1 = self.yIndexesMEC[0] if hmMec else self.yIndexesMEC[-1]
                            params = {'iY': iY1,
                                      'listBCInfo': self.__boundaryInfo(iY1, None, 'hmMec' if hmMec else bc),
                                      'hmRegCountOffset': self.hmRegionsIndex[hmCnt],
                                      'mecRegCountOffset': self.mecRegionsIndex[mecCnt],
                                      'removed_an': True if not hmMec and nextReg.split('_')[0] == 'vac' else False,
                                      'removed_bn': True if hmMec and prevReg.split('_')[0] == 'vac' else False,
                                      'lowerUpper': 'lower' if hmMec else 'upper'}
                            hmMec = not hmMec

                        for nHM in self.n:
                            params['nHM'] = nHM
                            # TODO We can try to cache these kind of functions for speed
                            getattr(self, bc)(**params)

                        # This conditional sets all the indices for the loop
                        self.__setCurrColCount(0, 0)
                        if bc != 'mecHm' or (bc == 'mecHm' and hmMec):
                            # Increment cnt for all hmHm boundaries
                            if bc == 'hmHm':
                                hmCnt += 1
                                if nextReg != 'mec' and nextnextReg.split('_')[0] != 'vac':
                                    cnt += 1
                            # Increment mecCnt only if leaving the mec region
                            if bc == 'mecHm' and hmMec:
                                cnt += 1
                                if nextReg.split('_')[0] != 'vac':
                                    nextY1 = yBoundaryIncludeMec[cnt] + 1
                                    cnt += 1
                                mecCnt += 1
                            if nextReg.split('_')[0] != 'vac':
                                iY1 = nextY1
                                plusOneVal = yBoundaryIncludeMec[cnt] + 1
                                nextY1 = plusOneVal

        print('Asize: ', self.matrixA.shape, self.matrixA.size)
        print('Bsize: ', self.matrixB.shape)
        print('(ppH, ppL, mecRegionLength)', f'({self.ppH}, {self.ppL}, {self.mecRegionLength})')

        self.__checkForErrors()

    def finalizeCompute(self):

        print('region indexes: ', self.hmRegionsIndex, self.mecRegionsIndex, self.mecRegionLength)

        self.__buildMatAB()

        # Solve for the unknown matrix X
        self.matrixX, preProcessError_matX = self.__linalg_lu()

        self.hmMatrixX = [self.matrixX[self.hmIdxs[i]] for i in range(len(self.hmIdxs))]
        self.mecMatrixX = [self.matrixX[self.mecIdxs[i]] for i in range(len(self.mecIdxs))]
        self.hmMatrixX = np.array(self.hmMatrixX, dtype=np.cdouble)
        self.mecMatrixX = np.array(self.mecMatrixX, dtype=np.cdouble)

        return preProcessError_matX

    def updateGrid(self, iErrorInX, canvasCfg, invertY=False):

        # Unknowns in HM regions
        matIdx = 0
        for i in self.hmRegions:
            # lower boundary
            if i == list(self.hmRegions)[0]:
                self.hmUnknownsList[i].an = self.hmMatrixX[:len(self.n)]
                matIdx += len(self.n)
            # upper boundary
            elif i == list(self.hmRegions)[-1]:
                self.hmUnknownsList[i].bn = self.hmMatrixX[-len(self.n):]
                matIdx += len(self.n)
            else:
                self.hmUnknownsList[i].an = self.hmMatrixX[matIdx: matIdx + 2 * len(self.n): 2]
                self.hmUnknownsList[i].bn = self.hmMatrixX[matIdx + 1: matIdx + 2 * len(self.n): 2]
                matIdx += 2 * len(self.n)

        # Unknowns in MEC regions
        for mecCnt, val in enumerate(self.mecRegions):
            Cnt = 0
            i, j = self.yIndexesMEC[0], 0
            while i < self.yIndexesMEC[-1] + 1:
                if i in self.yIndexesMEC:
                    while j < self.ppL:
                        self.matrix[i, j].Yk = self.mecMatrixX[Cnt]
                        Cnt += 1
                        j += 1
                    j = 0
                i += 1

        # Solve for B in the mesh
        i, j = 0, 0
        priorToCore, _ = self.getLastAndNextRegionName('mec')
        if self.allMecRegions or priorToCore.split('_')[0] == 'vac':
            regCnt = list(self.mecRegions)[0]
        else:
            regCnt = list(self.hmUnknownsList)[1:-1][0]
        while i < self.ppH:
            while j < self.ppL:
                lNode, rNode = self.neighbourNodes(j)
                if i in self.yIndexesMEC:
                    # Bottom layer of the MEC
                    if i == self.yIndexesMEC[0]:
                        if self.allMecRegions:
                            # Eqn 16
                            self.matrix[i, j].phiYp = self.__postEqn16to17(self.matrix[i + 1, j].Yk, self.matrix[i, j].Yk,
                                                                           self.matrix[i + 1, j].Ry, self.matrix[i, j].Ry)
                            # Eqn 17
                            self.matrix[i, j].phiYn = 0
                        else:
                            isNextToLowerVac = regCnt - 1 == list(self.hmUnknownsList)[0]
                            ur1, sigma1 = self.__getLowerUrSigma(i)
                            urSigma1 = ur1 * sigma1
                            nCnt, Flux_ySum = 0, 0
                            for nHM in self.n:

                                wn = 2 * nHM * pi / self.Tper
                                lambdaN1 = self.__lambda_n(wn, urSigma1)
                                if isNextToLowerVac:
                                    an = self.hmUnknownsList[regCnt - 1].an[nCnt]
                                    bn = 0
                                else:
                                    an = self.hmUnknownsList[regCnt - 1].an[nCnt]
                                    bn = self.hmUnknownsList[regCnt - 1].bn[nCnt]
                                Flux_ySum += self.__postEqn21(lambdaN1, wn, self.matrix[i, j].x,
                                                              self.matrix[i, j].x + self.matrix[i, j].lx,
                                                              self.matrix[i, j].y,
                                                              an, bn)
                                nCnt += 1

                            # Eqn 16
                            self.matrix[i, j].phiYp = self.__postEqn16to17(self.matrix[i + 1, j].Yk, self.matrix[i, j].Yk,
                                                                           self.matrix[i + 1, j].Ry, self.matrix[i, j].Ry)
                            # Eqn 17
                            self.matrix[i, j].phiYn = Flux_ySum

                    # Top layer of the MEC
                    elif i == self.yIndexesMEC[-1]:
                        if self.allMecRegions:
                            # Eqn 16
                            self.matrix[i, j].phiYp = 0
                            # Eqn 17
                            self.matrix[i, j].phiYn = self.__postEqn16to17(self.matrix[i, j].Yk, self.matrix[i - 1, j].Yk,
                                                                           self.matrix[i, j].Ry, self.matrix[i - 1, j].Ry)
                        else:
                            isNextToUpperVac = regCnt + 1 == list(self.hmUnknownsList)[-1]
                            ur2, sigma2 = self.__getUpperUrSigma(i)
                            urSigma2 = ur2 * sigma2
                            nCnt, Flux_ySum = 0, 0
                            for nHM in self.n:
                                wn = 2 * nHM * pi / self.Tper
                                lambdaN2 = self.__lambda_n(wn, urSigma2)
                                if isNextToUpperVac:
                                    an = 0
                                    bn = self.hmUnknownsList[regCnt + 1].bn[nCnt]
                                else:
                                    an = self.hmUnknownsList[regCnt + 1].an[nCnt]
                                    bn = self.hmUnknownsList[regCnt + 1].bn[nCnt]
                                Flux_ySum += self.__postEqn21(lambdaN2, wn, self.matrix[i, j].x,
                                                              self.matrix[i, j].x + self.matrix[i, j].lx,
                                                              self.matrix[i, j].y + self.matrix[i, j].ly,
                                                              an, bn)
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

                    self.matrix[i, j].phiError = self.matrix[i, j].phiXn + self.matrix[i, j].phiYn - self.matrix[i, j].phiXp - self.matrix[i, j].phiYp

                    # Eqn 40_HAM
                    self.matrix[i, j].Bx = self.postMECAvgB(self.matrix[i, j].phiXn, self.matrix[i, j].phiXp,
                                                            self.matrix[i, j].Szy)
                    self.matrix[i, j].By = self.postMECAvgB(self.matrix[i, j].phiYn, self.matrix[i, j].phiYp,
                                                            self.matrix[i, j].Sxz)

                elif i in self.yIndexesHM:
                    ur = self.matrix[i, 0].ur
                    sigma = self.matrix[i, 0].sigma
                    urSigma = ur * sigma
                    nCnt, BxSumCenter, BySumCenter = 0, 0, 0
                    for nHM in self.n:
                        wn = 2 * nHM * pi / self.Tper
                        lambdaN = self.__lambda_n(wn, urSigma)
                        an = self.hmUnknownsList[regCnt].an[nCnt]
                        bn = self.hmUnknownsList[regCnt].bn[nCnt]
                        BxSumCenter += self.__postEqn8(lambdaN, wn,
                                                       self.matrix[i, j].xCenter, self.matrix[i, j].yCenter, an, bn)
                        BySumCenter += self.__postEqn9(lambdaN, wn,
                                                       self.matrix[i, j].xCenter, self.matrix[i, j].yCenter, an, bn)

                        nCnt += 1

                    self.matrix[i, j].Bx = BxSumCenter
                    self.matrix[i, j].By = BySumCenter

                self.matrix[i, j].B = cmath.sqrt(self.matrix[i, j].Bx ** 2 + self.matrix[i, j].By ** 2)

                # Counter for each HM region
                if i in [val for val in self.yBoundaryList if val not in self.yIndexesMEC[:-1]] and j == self.ppL - 1:
                    regCnt += 1

                j += 1
            j = 0
            i += 1

        # This is a nested generator comprehension for phiError
        genPhiError = (j.phiError for i in self.matrix for j in i)
        postProcessError_Phi = max(genPhiError)
        genPhiError_min = (j.phiError for i in self.matrix for j in i)
        print(
            f'max error post processing is: {postProcessError_Phi} vs min error post processing: {min(genPhiError_min)} vs error in matX: {iErrorInX}')

        if postProcessError_Phi > self.errorTolerance or iErrorInX > self.errorTolerance:
            self.writeErrorToDict(key='name',
                                  error=Error.buildFromScratch(name='violatedKCL',
                                                               description="ERROR - Kirchhoff's current law is violated",
                                                               cause=True))

        # Thrust Calculation
        centerAirgap_y = self.matrix[self.yIdxCenterAirGap][0].yCenter

        ur = self.matrix[self.yIdxCenterAirGap, 0].ur
        sigma = self.matrix[self.yIdxCenterAirGap, 0].sigma

        resFx, resFy = np.cdouble(0), np.cdouble(0)
        thrustGenerator = self.__genForces(ur * sigma, centerAirgap_y)
        for (x, y) in thrustGenerator:
            resFx += x
            resFy += y

        resFx *= - j_plex * self.D * self.Tper / (2 * uo)
        resFy *= - self.D * self.Tper / (4 * uo)
        self.Fx = resFx
        self.Fy = resFy
        print(f'Fx: {round(self.Fx.real, 2)}N,', f'Fy: {round(self.Fy.real, 2)}N')

        if canvasCfg["showAirGapPlot"]:
            self.__plotPointsAlongX(self.yIdxCenterAirGap, invertY=invertY)
        if canvasCfg["showUnknowns"]:
            self.__plotPointsAlongHM(self.yIdxCenterAirGap)


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

    global row_FT, expandedLeftNodeEdges, slices, idx_FT, harmonics, model

    model = model_in

    idx_FT = 0
    harmonics = harmonics_in

    yIdx_lower = model.yIndexesMEC[0]
    yIdx_upper = model.yIndexesMEC[-1]
    row_FT = model.matrix[yIdx_lower]
    leftNodeEdges = [node.x for node in row_FT]
    slices = 5
    outerIdx, increment = 0, 0
    expandedLeftNodeEdges = list(np.zeros(len(leftNodeEdges) * slices, dtype=np.float64))
    for idx in range(len(expandedLeftNodeEdges)):
        if idx % slices == 0:
            increment = 0
            if idx != 0:
                outerIdx += 1
        else:
            increment += 1
        sliceWidth = row_FT[outerIdx].lx / slices
        expandedLeftNodeEdges[idx] = row_FT[outerIdx].x + increment * sliceWidth

    # noinspection PyGlobalUndefined
    def fluxAtBoundary():
        global row_FT, idx_FT, model

        lNode, rNode = model.neighbourNodes(idx_FT)
        phiXn = (row_FT[idx_FT].MMF + row_FT[lNode].MMF) \
                / (row_FT[idx_FT].Rx + row_FT[lNode].Rx)
        phiXp = (row_FT[idx_FT].MMF + row_FT[rNode].MMF) \
                / (row_FT[idx_FT].Rx + row_FT[rNode].Rx)
        return phiXn, phiXp

    # noinspection PyGlobalUndefined
    @lru_cache(maxsize=5)
    def pieceWise(x_in):
        global row_FT, idx_FT, model

        if x_in in leftNodeEdges[1:]:
            idx_FT += 1

        phiXn, phiXp = fluxAtBoundary()

        return model.postMECAvgB(phiXn, phiXp, row_FT[idx_FT].Szy)

    # noinspection PyGlobalUndefined
    @lru_cache(maxsize=5)
    def fourierSeries(x_in):
        global row_FT, idx_FT, model
        sumN = 0.0
        for nHM in harmonics:
            wn = 2 * nHM * pi / model.Tper
            # TODO *=2 if matching BC, correct without 2 though!
            #  we should try to multipy * 100 factor without 2 to see if it is closer
            coeff = j_plex / (wn * model.Tper)
            sumK = 0.0
            for iX in range(len(row_FT)):
                idx_FT = iX
                phiXn, phiXp = fluxAtBoundary()
                f = model.postMECAvgB(phiXn, phiXp, row_FT[iX].Szy)
                Xl = row_FT[iX].x
                Xr = Xl + row_FT[iX].lx
                resExp = cmath.exp(-j_plex * wn * (Xr - model.vel * model.t))\
                         - cmath.exp(-j_plex * wn * (Xl - model.vel * model.t))

                sumK += f * resExp

            sumN += coeff * sumK * cmath.exp(j_plex * wn * x_in)

        return sumN

    vfun = np.vectorize(pieceWise)
    idx_FT = 0
    x = expandedLeftNodeEdges
    y1 = vfun(x)
    vfun = np.vectorize(fourierSeries)
    y2 = vfun(x)

    # Background shading slots
    minY1, maxY1 = min(y2), max(y2)
    xPosLines = list(map(lambda x: x.x, model.matrix[0][model.slotArray[::model.ppSlot]]))
    # for cnt, each in enumerate(xPosLines):
    #     plt.axvspan(each, each + model.ws, facecolor='b', alpha=0.15, label="_" * cnt + "slot regions")
    # plt.legend()
    # plt.savefig('demo.png', transparent=True)

    plt.plot(x, y1, 'b-')
    plt.plot(x, y2, 'r-')
    plt.xlabel('Position [m]')
    plt.ylabel('Bx [T]')
    plt.title('Bx field at airgap Boundary')
    # plt.show()

    return x, y1


def plotFourierError():

    iterations = 1
    step = 2
    start = 6
    pixDivs = range(start, start + iterations * step, step)
    modelList = np.empty(len(pixDivs), dtype=ndarray)

    lowDiscrete = 50
    n = range(-lowDiscrete, lowDiscrete + 1)
    n = np.delete(n, len(n) // 2, 0)
    slots = 16
    poles = 6
    wt, ws = 6 / 1000, 10 / 1000
    slotpitch = wt + ws
    endTeeth = 2 * (5/3 * wt)
    length = ((slots - 1) * slotpitch + ws) + endTeeth
    canvasSpacing = 80

    for idx, pixelDivisions in enumerate(pixDivs):

        pixelSpacing = slotpitch / pixelDivisions
        regionCfg1 = {'hmRegions': {1: 'vac_lower', 2: 'bi', 3: 'dr', 4: 'g', 6: 'vac_upper'},
                      'mecRegions': {5: 'mec'},
                      'invertY': False}
        choiceRegionCfg = regionCfg1

        loopedModel = Model.buildFromScratch(slots=slots, poles=poles, length=length, n=n,
                                             pixelSpacing=pixelSpacing, canvasSpacing=canvasSpacing,
                                             hmRegions=choiceRegionCfg['hmRegions'],
                                             mecRegions=choiceRegionCfg['mecRegions'],
                                             errorTolerance=1e-15,
                                             # If invertY = False -> [LowerSlot, UpperSlot, Yoke]
                                             invertY=choiceRegionCfg['invertY'])

        loopedModel.buildGrid()
        loopedModel.checkSpatialMapping()
        modelList[idx] = ((pixelDivisions, loopedModel.ppL, loopedModel.ppH), complexFourierTransform(loopedModel, n))

    for (pixelDivisions, ppL, ppH), (xSequence, ySequence) in modelList:
        plt.plot(xSequence, ySequence, label=f'PixelDivs: {pixelDivisions}, (ppL, ppH): {(ppL, ppH)}')

    # plt.legend()
    plt.show()


if __name__ == '__main__':
    plotFourierError()
