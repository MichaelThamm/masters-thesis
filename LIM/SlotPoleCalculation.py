"""IMPORT LIBRARIES"""

import math
import cmath
import numpy as np
import contextlib
from timeit import default_timer as timer
from collections.abc import MutableMapping
import itertools
import configparser

# from numba import cuda, njit, int32, float64
# from numba.experimental import jitclass

"""
Constraints:

Pod Mass = 250kg
Top Speed = 500kph
Thrust = 2778N
L = 0.812m
Dmax = 0.067818m
J = 5A/mm2
Iin = 140Arms
FF = 0.6
q = integer values
g = 0.003m

Assumptions:
wt = 2/3 of the slot pitch
ws = 1/3 of the slot pitch


Filters:

"""


class LimMotor(object):
    def __init__(self, slots, poles, length, buildBaseline=False):

        self.errorDict = TransformedDict.buildFromScratch()

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

        # Conductivity
        self.sigma_iron = 4.5 * 10 ** 6  # Sm^-1
        self.sigma_alum = 17.0 * 10 ** 6  # Sm^-1
        self.sigma_air = 0  # Sm^-1
        self.sigma_copp = 5.96 * 10 ** 7  # Sm^-1

        # Permeability
        self.ur_iron = 1000
        self.ur_alum = 1
        self.ur_air = 1
        self.ur_copp = 1

        # Electrical Variables
        self.Ip = np.float64(10)  # AmpsPeak
        self.Jin = 0.0  # A/m^2
        self.vel = 0.0  # m/s
        self.f = 100.0  # Hz
        # self.f = self.vel/(2*self.Tp)  # Hz
        self.t = 0.0  # s
        FF = 0.6
        J = 5.0 * 10 ** 6  # A/m^2
        if buildBaseline:
            self.N = 57  # turns
        else:
            self.N = J * FF * (self.ws * self.hs / self.windingLayers) / self.Ip  # turns per coil

        self.Zeq = self.getImpedance()
        self.Vin = self.getVoltage()  # Volt

        # Stator Masses
        dCu = 8.96  # g/cm^3
        dSteel = 7.8  # g/cm^3
        dInsul = 1.4  # g/cm^3
        # temp_diamCond = min(currentDensityTable[0], key=lambda x: abs(x - self.Ip/math.sqrt(2)))
        temp_diamCond = np_find_nearest(currentDensityTable[0], self.Ip/math.sqrt(2))
        index_diamCond = np.where(currentDensityTable[0] == temp_diamCond)
        if index_diamCond[0].shape == (1,):
            idx_diamCond = index_diamCond[0][0]
        else:
            idx_diamCond = 0
            print('We have an issue in SlotPoleCalculation - defaulting to highest gauge wire')

        if self.Ip > currentDensityTable[0, index_diamCond]:
            diamCond = currentDensityTable[1, idx_diamCond-1]
        else:
            diamCond = currentDensityTable[1, idx_diamCond]
        MassCore = (self.hy*self.L + self.hs*(self.wt*(self.slots-1) + 2*self.endTooth)) * self.D * dSteel * 1000  # kg
        # TODO MassCu and MassInsul are not accurate if some slots are unfilled
        MassCu = 2*(self.m * self.slotpitch + self.D + 2*self.ws)*(diamCond/(10 ** 6))*self.N*self.slots*dCu*1000  # kg
        MassInsul = (1 - FF) * self.ws * self.hs * self.D * self.slots * dInsul * 1000  # kg
        self.MassTot = 2*(MassCore + MassCu + MassInsul)  # kg

    def writeErrorToDict(self, key, error):
        if error.state:
            self.errorDict.__setitem__(error.__dict__[key], error)

    def getDictKey(self):
        return self.errorDict.keys()

    def getInductance(self):
        # The inductance for a square coil is calculated as:
        # https://www.allaboutcircuits.com/tools/rectangle-loop-inductance-calculator/
        2*(self.m * self.slotpitch + self.D + 2*self.ws)
        coilLength = self.m * self.slotpitch + self.ws
        coilWidth = self.D + self.ws
        coilDiameter = 0
        sqExp = math.sqrt(coilLength ** 2 + coilWidth ** 2)
        term1 = 2 * (coilWidth + coilLength)
        term2 = 2 * sqExp
        term3 = coilLength * math.log((coilLength + sqExp)/coilWidth)
        term4 = coilWidth * math.log((coilWidth + sqExp)/coilLength)
        term5 = coilLength * math.log((2 * coilLength)/(coilDiameter/2))
        term6 = coilWidth * math.log((2 * coilWidth)/(coilDiameter/2))
        inductance = (uo * self.ur_iron / pi) * ((-term1) + term2 + (-term3) + (-term4) + term5 + term6)

    def getResistance(self):
        # I need to make sure this is correct in terms of total length and then think of it like the entire wire has
        # 4 loops
        coilLength = 2*(self.m * self.slotpitch + self.D + 2*self.ws) * self.N
        # area = diamCond/(10 ** 6)
        # resistance = (diamCond/(10 ** 6))*self.N*self.slots*dCu*1000
        pass

    def getImpedance(self):
        loops = 4
        # inductance = loops * (self.N ** 2 * uo * self.ur_iron * area / length)
        # the entire wire has 4 loops so it would be inductance summation

    def getVoltage(self):
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


pi = math.pi
j_plex = complex(0, 1)
uo = (10 ** - 7)*4*pi

# All Currents are in Amps and All cross sectional areas are in mm^2
# It is important to note that the cross sectional area is just for the bare copper and does not account for the insulation thickness
currentDensityTable = np.array([1060, 840, 665, 527.66, 418.47, 331.79, 263.12, 208.69, 165.44, 131.22,
104.11, 82.56, 65.44, 51.92, 41.13, 32.65, 25.92, 20.55, 16.3, 12.91,
10.26, 8.12, 6.45, 5.12, 4.06, 3.2, 2.56, 2.02, 1.6, 1.27,
1.01, 0.795, 0.64, 0.5, 0.395, 0.32, 0.251, 0.199, 0.158, 0.125,
0.099, 0.079, 0.062, 0.049, 0.039, 0.031, 0.025, 0.02, 0.016, 0.012,
0.01, 0.008, 0.006, 0.005, 136.77392, 108.38688, 85.80628, 68.08437996, 53.99602104, 42.81152728, 33.95089984, 26.92768808, 21.34705408, 16.93157904,
13.43352152, 10.65288192, 8.44320892, 6.69934144, 5.30708616, 4.21224964, 3.34450944, 2.65096244, 2.1032216, 1.66515796,
1.32386832, 1.04773984, 0.83161124, 0.66064384, 0.52386992, 0.4129024, 0.32967676, 0.26064464, 0.2064512, 0.16322548,
0.13032232, 0.10258044, 0.08258048, 0.064516, 0.05096764, 0.04129024, 0.032338645, 0.02564511, 0.02032254, 0.016129,
0.0127935228, 0.010129012, 0.0080451452, 0.0063806324, 0.00505934472, 0.00401225004, 0.00318192912, 0.00252322076, 0.00200128632, 0.0015870936,
0.00125870716, 0.00099806252, 0.00079161132, 0.00062774068], dtype=np.float64)
# currentDensityTable[0] = currents, currentDensityTable[1] = crossSections
currentDensityTable = currentDensityTable.reshape(2, int(currentDensityTable.size/2))

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
