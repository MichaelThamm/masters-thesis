"""IMPORT LIBRARIES"""

import math
import cmath
import numpy as np
import contextlib
from timeit import default_timer as timer
from collections.abc import MutableMapping

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
    def __init__(self, slots, poles, length):

        self.errorDict = TransformedDict.emptyDict()

        # Kinematic Variables
        mass = 250
        max_vel = 500*(1000/3600)
        acc_time = 12.5
        acceleration = max_vel/acc_time
        thrust = mass*acceleration
        power = thrust*max_vel
        synch_vel = max_vel

        # Stator Dimension Variables
        self.slots = slots
        self.poles = poles
        self.q = self.slots/self.poles/3
        self.L = length  # meters
        self.Tp = self.L/self.poles  # meters
        # self.slotpitch = self.L/self.slots  # meters
        # self.wt = 3/8*self.slotpitch  # meters
        # self.ws = self.slotpitch - self.wt  # meters
        self.wt = 6/1000  # meters
        self.ws = 10/1000  # meters
        self.slotpitch = self.ws + self.wt  # meters
        self.endTeeth = (self.L - ((self.slots - 1) * self.slotpitch + self.ws))/2  # meters
        # self.Tper = 12 * (self.slotpitch*3)  # meters
        self.Tper = 0.525  # meters

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
        self.N = 57  # turns
        self.f = 100.0  # Hz
        # self.f = self.vel/(2*self.Tp)  # Hz
        self.t = 0.0  # s
        FF = 0.6
        J = 5.0 * 10 ** 6  # A/m^2
        # This value is the number of turns for an entire slot even if 2 phases share the slot
        Nperslot = J*FF*self.ws*self.hs/self.Ip

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
        MassCore = (self.hy*self.L + self.wt*self.hs*self.slots) * self.D * dSteel * 1000  # kg
        MassCu = (self.Tp*2 + (self.D + self.ws)*2) * (diamCond/(10 ** 6)) * Nperslot * self.slots * dCu * 1000  # kg
        MassInsul = (1 - FF) * self.ws * self.hs * self.D * self.slots * dInsul * 1000  # kg
        self.MassTot = 2*(MassCore + MassCu + MassInsul)  # kg

    def writeErrorToDict(self, key, error):
        if error.state:
            self.errorDict.__setitem__(error.__dict__[key], error)

    def getDictKey(self):
        return self.errorDict.keys()


class TransformedDict(MutableMapping):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    @classmethod
    def emptyDict(cls):
        return cls()

    @classmethod
    def rebuildFromJson(cls, jsonDict):
        return cls(jsonDict)

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

    def printErrorsByAttr(self, strName):
        for key in self:
            print(self[key]['description'])

    def isEmpty(self):
        return False if self.store else False


class Error(object):
    def __init__(self, kwargs, buildFromJson=False):
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
