"""IMPORT LIBRARIES"""

import math
import cmath
import numpy as np
import contextlib
from timeit import default_timer as timer
from collections.abc import MutableMapping
import itertools
import configparser
import pandas as pd
import os

# from numba import cuda, njit, int32, float64
# from numba.experimental import jitclass


PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
pi = math.pi
j_plex = complex(0, 1)
uo = (10 ** - 7)*4*pi


class LimMotor(object):
    def __init__(self, slots, poles, length, buildBaseline=False):

        self.errorDict = TransformedDict.buildFromScratch()

        self.copper = Material(1, 8.96, 5.96 * 10 ** 7, 1.72 * 10 ** (-8))
        self.alum = Material(1, 8.96, 17.0 * 10 ** 6, None)
        self.air = Material(1, 8.96, 0, None)
        self.iron = Material(1000, 7.8, 4.5 * 10 ** 6, None)
        self.insul = Material(None, 1.4, None, None)

        # Kinematic Variables
        mass = 250
        max_vel = 500*(1000/3600)
        acc_time = 12.5
        acceleration = max_vel/acc_time
        thrust = mass*acceleration
        power = thrust*max_vel
        synch_vel = max_vel

        # Stator Dimension Variables
        self.m = 3
        self.slots = slots
        self.poles = poles
        self.q = self.slots/self.poles/self.m
        self.L = length  # meters
        self.Tp = self.L/self.poles  # meters

        if buildBaseline:
            self.ws = 10 / 1000  # meters
            self.wt = 6 / 1000  # meters
            self.Tper = 0.525  # meters
        else:
            self.ws = self.reverseWs(endTooth2SlotWidthRatio=1)  # meters
            self.wt = (3/5) * self.ws  # meters
            self.Tper = 1.25 * self.L  # meters  # TODO I should check with the baseline that a change to the Airbuffer does not make much difference to the result

        self.slotpitch = self.ws + self.wt  # meters
        self.endTooth = self.getLenEndTooth()  # meters
        self.writeErrorToDict(key='name',
                              error=Error.buildFromScratch(name='MotorLength',
                                                           description='ERROR - The inner dimensions do not sum to the motor length',
                                                           cause=not self.validateLength()))

        self.windingShift = 2
        self.windingLayers = 2

        self.Airbuffer = (self.Tper - self.L)/2  # meters
        self.hy = 6.5/1000  # meters
        self.hs = 20/1000  # meters
        self.dr = 2/1000  # meters
        self.g = 2.7/1000  # meters
        self.bi = 8/1000  # meters
        sfactor_D = 0.95
        self.D = 50/1000  # meters
        self.H = self.hy + self.hs  # meters
        self.vac = self.g * 1.5

        # Electrical Variables
        self.Ip = np.float64(10)  # AmpsPeak
        self.Jin = 0.0  # A/m^2
        self.vel = 0.0  # m/s
        self.f = 100.0  # Hz
        # self.f = self.vel/(2*self.Tp)  # Hz
        self.t = 0.0  # s
        fillFactor = 0.6
        J = 5.0 * 10 ** 6  # A/m^2
        if buildBaseline:
            self.N = 57  # turns
        else:
            self.N = J * fillFactor * (self.ws * self.hs / self.windingLayers) / self.Ip  # turns per coil
        self.coilLength = self.m * self.slotpitch
        self.coilWidth = self.D + self.ws

        self.removeUpperCoils = [0, self.slots-1, self.slots-2, self.slots-3]
        self.removeLowerCoils = [0, 1, 2, self.slots-1]

        closestCurrent = np_find_nearest(currentTable, self.Ip/math.sqrt(2))
        indexClosest = np.where(currentTable == closestCurrent)[0][0]
        if self.Ip > currentTable[indexClosest]:
            indexClosest -= 1
        self.areaConductor = areaTable[indexClosest] / (10 ** 6)  # m^2
        self.diamConductor = diamTable[indexClosest] / (10 ** 3)  # m
        self.currentConductor = currentTable[indexClosest]  # A

        self.loops = 4
        # Convert the square coil into a circular one to simplify inductance calculation
        self.diamLoop = (self.coilWidth + self.coilLength)
        self.perimeterCoil = pi * self.diamLoop
        volCu = 2 * self.perimeterCoil * self.N * self.loops * self.areaConductor  # m^2
        volInsul = (1 - fillFactor) * (volCu / fillFactor)  # m^2
        massCu = volCu * self.copper.density*1000  # kg
        massInsul = volInsul * self.insul.density*1000  # kg
        massCore = (self.hy*self.L + self.hs*(self.wt*(self.slots-1) + 2*self.endTooth))*self.D*self.iron.density*1000  # kg
        self.massTot = massCore + massCu + massInsul  # kg

        self.Res = self.getResistancePerPhase()
        # TODO Use the LIM_EquivalentCircuit paper to calculate this
        self.Zeq = self.getImpedancePerPhase()
        self.Vin = self.getVoltagePerPhase()  # Volt

    def writeErrorToDict(self, key, error):
        if error.state:
            self.errorDict.__setitem__(error.__dict__[key], error)

    def getDictKey(self):
        return self.errorDict.keys()

    def getInductancePerPhase(self):
        # The inductance for a square coil is calculated as:
        # https://www.allaboutcircuits.com/tools/rectangle-loop-inductance-calculator/
        coilRadius = self.diamConductor / 2  # meters
        areaCoilCircle = pi * coilRadius ** 2
        return uo * self.N ** 2 * areaCoilCircle

    def getResistancePerPhase(self):
        phaseWindingLength = self.perimeterCoil * self.N * self.loops
        return self.copper.resistivity * phaseWindingLength / self.areaConductor

    def getImpedancePerPhase(self):
        Figure out equivalent circuit model
        impedancePrimary =
        impedanceSecondary =
        resistanceSecondary =
        loops = 4
        # inductance = loops * (self.N ** 2 * uo * self.ur_iron * area / length)
        # the entire wire has 4 loops so it would be inductance summation

    def getVoltagePerPhase(self):
        pass
        # self.Zeq = self.getResistance() + j_plex * 2 * pi * self.f * self.getInductance()
        # self.V =

    def reverseWs(self, endTooth2SlotWidthRatio):
        # In general, endTooth2SlotWidthRatio should be 1 so the end teeth are the same size as slot width
        coeff = (8/5) * (self.slots - 1) + 1 + 2 * endTooth2SlotWidthRatio
        return self.L / coeff

    def getLenEndTooth(self):
        return (self.L - (self.slotpitch * (self.slots - 1) + self.ws)) / 2

    def validateLength(self):
        return round(self.L - (self.slotpitch*(self.slots-1)+self.ws+2*self.endTooth), 12) == 0


class Material(object):
    def __init__(self, ur, density, sigma, resistivity):
        self.ur = ur
        self.density = density  # g/cm^3
        self.sigma = sigma  # Sm^-1
        self.resistivity = resistivity  # OhmMeter


class TransformedDict(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            self.store = dict()
            for key, error in kwargs.items():
                self.__setitem__(key, Error.buildFromJson(error))
            return

        self.store = dict()
        self.update(dict(kwargs))  # use the free update to set keys

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    def __eq__(self, otherObject):
        if not isinstance(otherObject, TransformedDict):
            # don't attempt to compare against unrelated types
            return NotImplemented
        # If the objects are the same then set the IDs to be equal
        elif self.__dict__.items() == otherObject.__dict__.items():
            for attr, val in otherObject.__dict__.items():
                return self.__dict__[attr] == otherObject.__dict__[attr]
        # The objects are not the same
        else:
            pass

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def genStoreByValueAttr(self, strName):
        return (self.store[strName] for _ in range(self.store.__len__()))

    def printErrorsByAttr(self, attrString):
        for key in self.store:
            print(self.__getitem__(key).__dict__[attrString])

    def isEmpty(self):
        return False if self.store else False


class Error(object):
    def __init__(self, kwargs, buildFromJson=False):

        if buildFromJson:
            for key in kwargs:
                self.__dict__[key] = kwargs[key]
            return

        self.name = kwargs['name']
        self.description = kwargs['description']
        self.cause = bool(kwargs['cause'])  # This was done to handle np.bool_ not being json serializable
        self.state = False

        self.setState()

    @classmethod
    def buildFromScratch(cls, **kwargs):
        return cls(kwargs=kwargs)

    @classmethod
    def buildFromJson(cls, jsonObject):
        return cls(kwargs=jsonObject, buildFromJson=True)

    def __eq__(self, otherObject):
        if not isinstance(otherObject, Error):
            # don't attempt to compare against unrelated types
            return NotImplemented
        # If the objects are the same then set the IDs to be equal
        elif self.__dict__.items() == otherObject.__dict__.items():
            for attr, val in otherObject.__dict__.items():
                return self.__dict__[attr] == otherObject.__dict__[attr]
        # The objects are not the same
        else:
            pass

    def setState(self):
        if self.cause:
            self.state = True
        else:
            self.state = False


def np_find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


@contextlib.contextmanager
def timing():
    s = timer()
    yield
    e = timer()
    print('execution time: {:.2f}s'.format(e - s))


def rebuildPlex(val):
    try:
        if val[0] == 'plex_Signature':
            return np.cdouble(val[1] + j_plex * val[2])
    except TypeError:
        return val


def getCurrDensityTable(key):
    """
    Note: The values in the WireSpecs sheet are not accounting for insulation
    Note: All current is in amps
    """
    xlsxFile = os.path.join(PROJECT_PATH, 'SupportingDocs\\Calculators\\CurrentDensityChart.xlsx')
    xl_file = pd.ExcelFile(xlsxFile)

    dfs = {sheet_name: xl_file.parse(sheet_name)
           for sheet_name in xl_file.sheet_names}

    return dfs[key]


currentTable = getCurrDensityTable('WireSpecs')['Current']
diamTable = getCurrDensityTable('WireSpecs')['Dia in mm ']
areaTable = getCurrDensityTable('WireSpecs')['Area in mm.sq']

matList = np.array([('iron', 'gray10'), ('copperA', 'red'), ('copperB', 'green'), ('copperC', 'DodgerBlue2'),
                    ('aluminum', '#717676'), ('vacuum', '#E4EEEE')])

# table.sort(reverse=True)
# Sort(table)

'''
# If file already exists in the current directory
if os.path.isfile('BuildTable.csv'):
    os.unlink('BuildTable.csv')

# Write to the file
with open('BuildTable.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['kw', 'slots', 'poles', 'q'])
    for value in table:
        writer.writerow(value)
'''

if __name__ == '__main__':
    pass
