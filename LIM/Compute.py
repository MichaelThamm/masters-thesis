from LIM.Grid import *
from LIM.SlotPoleCalculation import np
from scipy.linalg import lu_factor, lu_solve
import matplotlib.pyplot as plt
from functools import lru_cache


# This performs the lower-upper decomposition of A to solve for x in Ax = B
# A must be a square matrix
# @njit("float64[:](float64[:, :], float64[:])", cache=True)
class Model(Grid):

    currColCount, currColCountUpper, nLoop, matBCount = 0, 0, 0, 0

    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for attr_key in kwargs:
                self.__dict__[attr_key] = kwargs[attr_key]
            return

        super().__init__(kwargs)

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='meshDensityDiscrepancy',
                                                           description="ERROR - The last slot has a different mesh density than all other slots",
                                                           cause=kwargs['meshIndexes'][0][2] != kwargs['meshIndexes'][0][-3]))

        self.errorTolerance = kwargs['errorTolerance']

        self.hmUnknownsList = {}
        self.canvasRowRegIdxs, self.canvasColRegIdxs = [], []
        self.mecIdxs = []
        self.hmMatrixX, self.mecMatrixX = [], []
        self.vacLowerRow, self.vacUpperRow = [], []

        lenUnknowns = self.hmRegionsIndex[-1]

        self.matrixA = np.zeros((lenUnknowns, lenUnknowns), dtype=np.cdouble)
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
            if region != 'core' and region.split('_')[0] != 'vac':
                indexColVal += 2 * len(self.n)
                indexRowVal += 2 * len(self.n)
                self.canvasRowRegIdxs.append(indexRowVal)

            elif region.split('_')[0] == 'vac':
                indexColVal += len(self.n)

            elif region == 'core':
                indexColVal += self.mecRegionLength
                for bc in self.getFullRegionDict()[region]['bc'].split(', '):
                    if bc == 'mecHm':
                        indexRowVal += len(self.n)
                    elif bc == 'mec':
                        indexRowVal += self.mecRegionLength

                    self.canvasRowRegIdxs.append(indexRowVal)

            self.canvasColRegIdxs.append(indexColVal)
        self.mecCanvasRegIdxs = [self.canvasRowRegIdxs[list(self.getFullRegionDict()).index('core')-1] + self.ppL * i for i in range(1, self.ppHeight)]

        for i in self.mecRegionsIndex:
            self.mecIdxs.extend(list(range(i, i + self.mecRegionLength)))
        self.hmIdxs = [i for i in range(len(self.matrixX)) if i not in self.mecIdxs]

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
                # Check keys are equal
                if entry[0] == otherItems[cnt][0]:
                    # np.ndarray were converted to lists to dump data to json object, now compare against those lists
                    if type(entry[1]) == np.ndarray:
                        # Greater than 1D array
                        if len(entry[1].shape) > 1:
                            # Check that 2D array are equal
                            if np.all(np.all(entry[1] != otherItems[cnt][1], axis=1)):
                                equality = False
                        # 1D array
                        else:
                            if entry[1].tolist() != otherItems[cnt][1]:
                                # The contents of the array or list may be a deconstructed complex type
                                if np.all(list(map(lambda val: val[1] == rebuildPlex(otherItems[cnt][1][val[0]]), enumerate(entry[1])))):
                                    pass
                                else:
                                    equality = False
                    else:
                        if entry[1] != otherItems[cnt][1]:
                            # Check to see if keys of the value are equal to string versions of otherItems keys
                            try:
                                if list(map(lambda x: str(x), entry[1].keys())) != list(otherItems[cnt][1].keys()):
                                    equality = False
                            except ValueError:
                                pass
                else:
                    equality = False

        return equality

        self.currColCount += 2
        self.nLoop += 1
        self.matBCount += 1

    def hmHm(self, nHM, listBCInfo, RegCountOffset1, RegCountOffset2, remove_an, remove_bn):

        hb1, ur1, urSigma1, _, ur2, urSigma2 = listBCInfo

        wn = 2 * nHM * pi / self.Tper
        lambdaN1 = self.__lambda_n(wn, urSigma1)
        lambdaN2 = self.__lambda_n(wn, urSigma2)

        # By Condition
        Ba_lower, Ba_upper, Bb_lower, Bb_upper = self.__preEqn24_2018(lambdaN1, lambdaN2, hb1)
        self.matrixA[self.nLoop, RegCountOffset1 + self.currColCount] = Ba_lower  # an Lower
        if not remove_bn:
            self.matrixA[self.nLoop, RegCountOffset1 + 1 + self.currColCount] = Bb_lower  # bn Lower
        if not remove_an:
            self.matrixA[self.nLoop, RegCountOffset2 + self.currColCountUpper] = Ba_upper  # an Upper
        self.matrixA[self.nLoop, RegCountOffset2 + 1 + self.currColCountUpper] = Bb_upper  # bn Upper

        # Hx Condition
        Ha_lower, Ha_upper, Hb_lower, Hb_upper = self.__preEqn25_2018(lambdaN1, lambdaN2, ur1, ur2, hb1)

        self.matrixA[self.nLoop + 1, RegCountOffset1 + self.currColCount] = Ha_lower  # an Lower
        if not remove_bn:
            self.matrixA[self.nLoop + 1, RegCountOffset1 + 1 + self.currColCount] = Hb_lower  # bn Lower
        if not remove_an:
            self.matrixA[self.nLoop + 1, RegCountOffset2 + self.currColCountUpper] = Ha_upper  # an Upper
        self.matrixA[self.nLoop + 1, RegCountOffset2 + 1 + self.currColCountUpper] = Hb_upper  # bn Upper

        self.incrementCurrColCnt(remove_an, remove_bn)
        self.nLoop += 2
        self.matBCount += 2

    def mecHm(self, nHM, iY, listBCInfo, hmRegCountOffset, mecRegCountOffset, removed_an, removed_bn, lowerUpper):
        hb, ur, urSigma = listBCInfo
        wn = 2 * nHM * pi / self.Tper
        lambdaN = self.__lambda_n(wn, urSigma)
        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)
        # TODO My math shows this to be 1 2019 paper says 2
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
                self.matrixA[self.nLoop, mecRegCountOffset + iX] = combinedPsi_k  # Curr
            elif lowerUpper == 'upper':
                self.matrixA[self.nLoop, mecRegCountOffset + self.mecRegionLength - self.ppL + iX] = combinedPsi_k  # Curr

            sumResEqn22Source += (resSource_kn + resSource_k + resSource_kp)

        # HM related equations
        hmResA, hmResB = self.__preEqn8(ur, lambdaN, wn, hb)

        # If the neighbouring region is a Dirichlet region, an or bn may be removed
        # a not b
        if not removed_an and removed_bn:
            self.matrixA[self.nLoop, hmRegCountOffset + self.currColCount] = hmResA  # an
        # b not a
        elif removed_an and not removed_bn:
            self.matrixA[self.nLoop, hmRegCountOffset + self.currColCountUpper] = hmResB  # bn
        # a and b
        elif not removed_an and not removed_bn:
            self.matrixA[self.nLoop, hmRegCountOffset + self.currColCount] = hmResA  # an
            self.matrixA[self.nLoop, hmRegCountOffset + 1 + self.currColCount] = hmResB  # bn
        # a and b
        else:
            print('There was an error here')
            return None

        self.matrixB[self.matBCount] = - coeff * sumResEqn22Source

        self.incrementCurrColCnt(removed_an, removed_bn)
        self.nLoop += 1
        self.matBCount += 1

    def mec(self, i, j, node, time_plex, listBCInfo,
            hmRegCountOffset1, hmRegCountOffset2, mecRegCountOffset, removed_an, removed_bn):

        hb1, urSigma1, hb2, urSigma2 = listBCInfo

        lNode, rNode = self.neighbourNodes(j)
        if i == 0:
            _, vacResY = self.matrix[i, j].getReluctance(vacHeight=self.vac/self.ppVac, isVac=True)
            northRelDenom = self.matrix[i, j].Ry + self.matrix[i + 1, j].Ry
            southRelDenom = self.matrix[i, j].Ry + vacResY
        elif i == self.ppH - 1:
            _, vacResY = self.matrix[i, j].getReluctance(vacHeight=self.vac/self.ppVac, isVac=True)
            northRelDenom = self.matrix[i, j].Ry + vacResY
            southRelDenom = self.matrix[i, j].Ry + self.matrix[i - 1, j].Ry
        else:
            northRelDenom = self.matrix[i, j].Ry + self.matrix[i + 1, j].Ry
            southRelDenom = self.matrix[i, j].Ry + self.matrix[i - 1, j].Ry
        eastRelDenom = self.matrix[i, j].Rx + self.matrix[i, rNode].Rx
        westRelDenom = self.matrix[i, j].Rx + self.matrix[i, lNode].Rx

        eastMMFNum = self.matrix[i, j].MMF + self.matrix[i, rNode].MMF
        westMMFNum = self.matrix[i, j].MMF + self.matrix[i, lNode].MMF

        # Bottom layer of the mesh
        if i == self.yIndexesMEC[0]:

            self.__setCurrColCount(0, 0)
            # South Node
            for nHM in self.n:
                wn = 2 * nHM * pi / self.Tper
                lambdaN1 = self.__lambda_n(wn, urSigma1)
                anCoeff, bnCoeff = self.__preEqn21(lambdaN1, wn, self.matrix[i, j].x,
                                                   self.matrix[i, j].x + self.matrix[i, j].lx, hb1)

                self.matrixA[self.nLoop + node, hmRegCountOffset1 + self.currColCount] = anCoeff  # an
                if not removed_bn:
                    self.matrixA[self.nLoop + node, hmRegCountOffset1 + 1 + self.currColCount] = bnCoeff  # bn

                self.incrementCurrColCnt(False, removed_bn)

            # North Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node + self.ppL] = - time_plex / northRelDenom

            # Current Node
            self.matrixA[self.nLoop + node, mecRegCountOffset + node] = time_plex * (1 / westRelDenom + 1 / eastRelDenom + 1 / northRelDenom)

        # Top layer of the mesh
        elif i == self.yIndexesMEC[-1]:

            self.__setCurrColCount(0, 0)
            # North Node
            for nHM in self.n:
                wn = 2 * nHM * pi / self.Tper
                lambdaN2 = self.__lambda_n(wn, urSigma2)
                anCoeff, bnCoeff = self.__preEqn21(lambdaN2, wn, self.matrix[i, j].x,
                                                   self.matrix[i, j].x + self.matrix[i, j].lx, hb2)

                if removed_an:
                    self.matrixA[self.nLoop + node, hmRegCountOffset2 + self.currColCountUpper] = - bnCoeff  # bn
                else:
                    self.matrixA[self.nLoop + node, hmRegCountOffset2 + self.currColCountUpper] = - anCoeff  # an
                    self.matrixA[self.nLoop + node, hmRegCountOffset2 + 1 + self.currColCountUpper] = - bnCoeff  # bn

                self.incrementCurrColCnt(removed_an, False)

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
                                                           description='Error - nLoop',
                                                           cause=self.nLoop != self.matrixA.shape[0]))

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='matBCount',
                                                           description='Error - matBCount',
                                                           cause=self.matBCount != self.matrixB.size))

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='regCount',
                                                           description='Error - regCount',
                                                           cause=self.hmRegionsIndex[-1] != self.matrixA.shape[1]))

        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='regCount',
                                                           description='Error - The matrices A and B do not match',
                                                           cause=self.matrixA.shape[0] != self.matrixB.size))

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
            ur1 = self.matrix[iY1 - 1, 0].ur if iY1 != 0 else self.ur_air
            sigma1 = self.matrix[iY1 - 1, 0].sigma if iY1 != 0 else self.sigma_air
            urSigma1 = ur1 * sigma1
            hb2 = self.matrix[iY2+1, 0].y if iY2 != self.ppH - 1 else self.modelHeight
            ur2 = self.matrix[iY2+1, 0].ur if iY2 != self.ppH - 1 else self.ur_air
            sigma2 = self.matrix[iY2+1, 0].sigma if iY2 != self.ppH - 1 else self.sigma_air
            urSigma2 = ur2 * sigma2

            return [hb1, urSigma1, hb2, urSigma2]

        elif boundaryType == 'hmHm' and not iY2:
            hb1 = self.matrix[iY1, 0].y
            ur1 = self.matrix[iY1 - 1, 0].ur if iY1 != 0 else self.ur_air
            sigma1 = self.matrix[iY1 - 1, 0].sigma if iY1 != 0 else self.sigma_air
            urSigma1 = ur1 * sigma1
            ur2 = self.matrix[iY1, 0].ur
            sigma2 = self.matrix[iY1, 0].sigma
            urSigma2 = ur2 * sigma2

            return [hb1, ur1, urSigma1, None, ur2, urSigma2]

        elif boundaryType == 'hmMec' and not iY2:
            hb = self.matrix[iY1, 0].y
            ur = self.matrix[iY1 - 1, 0].ur if iY1 != 0 else self.ur_air
            sigma = self.matrix[iY1 - 1, 0].sigma if iY1 != 0 else self.sigma_air
            urSigma = ur * sigma

            return [hb, ur, urSigma]

        elif boundaryType == 'mecHm' and not iY2:
            hb = self.matrix[iY1 + 1, 0].y if iY1 != self.ppH - 1 else self.modelHeight
            ur = self.matrix[iY1 + 1, 0].ur if iY1 != self.ppH - 1 else self.ur_air
            sigma = self.matrix[iY1 + 1, 0].sigma if iY1 != self.ppH - 1 else self.sigma_air
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

    def __plotPointsAlongX(self, evenOdd, iY):

        # TODO Although I now have the correct y values, I have to check the right hand rule on the positive x and y directions
        #  if they axis do not follow right hand rule then the graph will be flipped.
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

        dataArray[1] = yBxList
        dataArray[2] = yByList
        dataArray[3] = yB_List

        # TODO This inverts the y axis for the plot
        dataArray[1] = np.flip(yBxList)
        dataArray[2] = np.flip(yByList)

        xSorted = np.array([i.real for i in dataArray[0]], dtype=np.float64)

        lineWidth = 2
        markerSize = 5
        tempReal = np.array([j.real for j in dataArray[1]], dtype=np.float64)
        plt.scatter(xSorted, tempReal.flatten())
        plt.plot(xSorted, tempReal.flatten(), marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('Position [m]')
        plt.ylabel('Bx [T]')
        plt.title('Bx field in airgap')
        # ax = plt.gca()
        # ax.set_ylim(ax.get_ylim()[::-1])
        # ax.invert_yaxis()
        plt.show()

        tempReal = np.array([j.real for j in dataArray[2]], dtype=np.float64)
        plt.scatter(xSorted, tempReal.flatten())
        plt.plot(xSorted, tempReal.flatten(), marker='o', linewidth=lineWidth, markersize=markerSize)
        plt.xlabel('Position [m]')
        plt.ylabel('By [T]')
        plt.title('By field in airgap')
        # ax = plt.gca()
        # ax.set_ylim(ax.get_ylim()[::-1])
        # ax.invert_yaxis()
        plt.show()

    def __buildMatAB(self):

        lenUnknowns = self.hmRegionsIndex[-1]
        time_plex = cmath.exp(j_plex * 2 * pi * self.f * self.t)

        yBoundaryIncludeMec = [index for index in self.yBoundaryList if index not in self.yIndexesMEC]
        yBoundaryIncludeMec.extend([self.yIndexesMEC[0], self.yIndexesMEC[-1]])
        yBoundaryIncludeMec.sort()

        # TODO There is potential here to use concurrent.futures.ProcessPoolExecutor since the indexing of the matrix A does not depend on previous boundary conditions
        # TODO This will require me to change the way I pass the nLoop and matBCount from boundary condition to boundary condition. Instead these need to be constants
        hmMec = True
        cnt, hmCnt, mecCnt = 0, 0, 0
        iY1, nextY1 = 0, yBoundaryIncludeMec[0] + 1
        for region in self.getFullRegionDict():
            if region.split('_')[0] != 'vac':
                prevReg, nextReg = self.getLastAndNextRegionName(region)
                for bc in self.getFullRegionDict()[region]['bc'].split(', '):

                    # Mec region calculation
                    if bc == 'mec':
                        node = 0
                        iY1, iY2 = self.yIndexesMEC[0], self.yIndexesMEC[-1]
                        i, j = iY1, 0
                        while i < iY2 + 1:
                            while j < self.ppL:
                                params = {'i': i, 'j': j, 'node': node, 'time_plex': time_plex,
                                          'listBCInfo': self.__boundaryInfo(iY1, iY2, bc),
                                          'hmRegCountOffset1': self.hmRegionsIndex[hmCnt],
                                          'hmRegCountOffset2': self.hmRegionsIndex[hmCnt+1],
                                          'mecRegCountOffset': self.mecRegionsIndex[mecCnt],
                                          'removed_an': True if nextReg.split('_')[0] == 'vac' else False,
                                          'removed_bn': True if prevReg.split('_')[0] == 'vac' else False}

                                # TODO We can try to cache these kind of functions for speed
                                getattr(self, bc)(**params)

                                node += 1
                                j += 1
                            j = 0
                            i += 1
                        # Increment the indexing after finishing with the mec region
                        self.__setCurrColCount(0, 0)
                        self.nLoop += node
                        self.matBCount += node
                        nextY1 = iY2 + 1
                        cnt += 1
                        hmCnt += 1

                    # All boundary conditions loop through N harmonics except mec
                    else:
                        # Loop through harmonics and calculate each boundary condition
                        if bc == 'hmHm':
                            # The problem is here since iY1 is 8 because we didnt skip the first hmHm bound. Could check nextReg, prevReg
                            # I need to make sure my solution is robust for Cfg1 as well. Once I have tested Cfg1 and Cfg2
                            # I need to merge removeDirichlet and invertTK. Then I can create rows for self.vacLowerRow and UpperRow
                            # I will need to jump into a couple of methods throughout Grid and Compute to build it,
                            # otherwise the row nodes will not have accurate values for all attributes like Rx, Ry
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
                                if nextReg != 'core':
                                    cnt += 1
                            # Increment mecCnt only if leaving the mec region
                            if bc == 'mecHm' and hmMec:
                                cnt += 1
                                mecCnt += 1
                            iY1 = nextY1
                            nextY1 = yBoundaryIncludeMec[cnt] + 1

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

    def updateGrid(self, iErrorInX, showAirgapPlot=False):

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
        for mecCnt in range(len(self.mecRegions)):
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
        regCnt = list(self.hmUnknownsList)[1:-1][0]
        while i < self.ppH:
            while j < self.ppL:

                lNode, rNode = self.neighbourNodes(j)

                if i in self.yIndexesMEC:

                    # Bottom layer of the MEC
                    if i == self.yIndexesMEC[0]:
                        ur1 = self.ur_air if i == 0 else self.matrix[i - 1, 0].ur
                        sigma1 = self.sigma_air if i == 0 else self.matrix[i - 1, 0].sigma
                        urSigma1 = ur1 * sigma1
                        nCnt, Flux_ySum = 0, 0
                        for nHM in self.n:

                            wn = 2 * nHM * pi / self.Tper
                            lambdaN1 = self.__lambda_n(wn, urSigma1)
                            # TODO Here - We still need to test this for Cfg2
                            isNextToLowerVac = regCnt - 1 == list(self.hmUnknownsList)[0]
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
                        ur2 = self.ur_air if i == self.ppH - 1 else self.matrix[i + 1, 0].ur
                        sigma2 = self.sigma_air if i == self.ppH - 1 else self.matrix[i + 1, 0].sigma
                        urSigma2 = ur2 * sigma2
                        nCnt, Flux_ySum = 0, 0
                        for nHM in self.n:
                            wn = 2 * nHM * pi / self.Tper
                            lambdaN2 = self.__lambda_n(wn, urSigma2)
                            # TODO Here - We still need to test this for Cfg2
                            isNextToUpperVac = regCnt + 1 == list(self.hmUnknownsList)[-1]
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
                if i in [val for val in self.yBoundaryList if val not in self.yIndexesMEC[:-1]] and j == self.ppL - 1:
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

        if postProcessError_Phi > self.errorTolerance or iErrorInX > self.errorTolerance:
            self.writeErrorToDict(key='name',
                                  error=Error.buildFromScratch(name='violatedKCL',
                                                               description="ERROR - Kirchhoff's current law is violated",
                                                               cause=True))

        # Thrust Calculation
        centerAirgapIdx_y = self.yIndexesAirgap[0] + self.ppAirGap // 2
        if self.ppAirGap % 2 == 0:  # even
            evenOdd = 'even'
            centerAirgap_y = self.matrix[centerAirgapIdx_y][0].y
        else:  # odd
            evenOdd = 'odd'
            centerAirgap_y = self.matrix[centerAirgapIdx_y][0].yCenter

        ur = self.matrix[centerAirgapIdx_y, 0].ur
        sigma = self.matrix[centerAirgapIdx_y, 0].sigma

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

    # TODO Clean this code up - not working since Cfg1 and Cfg2 changes
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
    # plt.plot(x, y1, 'b-')
    # plt.plot(x, y2, 'r-')
    # TODO 'Be aware that you have to set the axis limits before you invert the axis, otherwise it will un-invert it again.'
    # TODO Test the inversion
    # plt.gca().invert_yaxis()
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

    'Be aware that you have to set the axis limits before you invert the axis, otherwise it will un-invert it again.'
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # profile_main()  # To profile the main execution
    plotFourierError()