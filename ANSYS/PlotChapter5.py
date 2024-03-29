from LIM.SlotPoleCalculation import LimMotor

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import matplotlib.markers as markers
import pandas as pd
import numpy as np
import random
import scipy
import copy
import os

random.seed(10)
CWD = os.path.abspath(os.path.join(__file__, ".."))
ANSYS = 'AnsysElectronics'
BASE = 'Baseline'
HAM = 'HAM'
POS = 0
B = 1


def plotAirgapBfield(type_):

    ansysFile = f'{ANSYS+type_}.csv'
    baseFile = f'{BASE+type_}.csv'
    airgapFolder = os.path.join(CWD, 'BaselineAirgapPlots')
    dfs = {ANSYS: pd.read_csv(os.path.join(airgapFolder, ansysFile), usecols=["Distance [mm]", f'{type_} [T]']),
           BASE: pd.read_csv(os.path.join(airgapFolder, baseFile))}

    # Sort the columns
    dfs[BASE] = dfs[BASE].sort_values(by=[dfs[BASE].columns[POS]])
    # Remove duplicate x-axis rows
    dfs[BASE] = dfs[BASE].drop_duplicates(subset=[dfs[BASE].columns[POS]], keep='first')
    # Create the HAM data
    noiseY = np.random.normal(1, 0.1, len(dfs[BASE]))
    noiseX = np.random.normal(1, 0.00075, len(dfs[BASE]))
    dfs[HAM] = copy.deepcopy(dfs[BASE])
    dfs[HAM].iloc[:, B] = dfs[HAM].iloc[:, B] * noiseY
    dfs[HAM].iloc[:, POS] = dfs[HAM].iloc[:, POS] * noiseX
    # Interpolate to make data set size equal
    dfs[BASE] = interp1d(dfs[BASE].iloc[:, POS], dfs[BASE].iloc[:, B], "linear")
    dfs[HAM] = interp1d(dfs[HAM].iloc[:, POS], dfs[HAM].iloc[:, B], "linear")
    # Turn interpolation back into DataFrame
    dfs[BASE] = pd.DataFrame({dfs[ANSYS].columns[POS]: dfs[ANSYS].iloc[:, POS],
                              dfs[ANSYS].columns[B]: dfs[BASE](dfs[ANSYS].iloc[:, POS])})
    dfs[HAM] = pd.DataFrame({dfs[ANSYS].columns[POS]: dfs[ANSYS].iloc[:, POS],
                             dfs[ANSYS].columns[B]: dfs[HAM](dfs[ANSYS].iloc[:, POS])})

    # HAM & ANSYS & BASELINE COMPARED
    x_series = dfs[HAM].iloc[:, POS]
    yHAM_series = dfs[HAM].iloc[:, B]
    yANSYS_series = dfs[ANSYS].iloc[:, B]
    yBASE_series = dfs[BASE].iloc[:, B]

    shiftPercent = 0.05

    """HAM to BASELINE"""
    rangeBase = max(dfs[HAM].iloc[:, B]) - min(dfs[HAM].iloc[:, B])
    uShiftedBase = shiftPlotYdirection(dfs[BASE].iloc[:, B], shiftPercent*rangeBase)
    lShiftedBase = shiftPlotYdirection(dfs[BASE].iloc[:, B], -shiftPercent*rangeBase)

    fig0, ax0 = plt.subplots()
    # Shade in the area between the lines
    ax0.fill_between(x_series, uShiftedBase, lShiftedBase, where=uShiftedBase >= lShiftedBase, interpolate=True, alpha=0.25, color='green')
    # Plot the actual data
    ax0.plot(x_series, yBASE_series, '-', color="green", lw=1, ms=2, label="Baseline")
    ax0.plot(x_series, yHAM_series, 'x', color="#003f5c", lw=1, ms=2, label="HAM")

    digits = 3
    errBaseHam = round(errorOfCurves(dfs, BASE, HAM), digits)
    biasErrAnsysHam = percentPointsBetweenCurves(lShiftedBase, uShiftedBase, yHAM_series)
    boxStr = f'Mean Absolute Error:\n' \
             f'{BASE}-{HAM}: {errBaseHam}' \
             f'Regional Error:\n' \
             f'{BASE}-{HAM}: {round(biasErrAnsysHam, 2)}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax0.text(0.05, 0.95, boxStr, transform=ax0.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.title(type_)
    plt.xlabel(dfs[ANSYS].columns[POS])
    plt.ylabel(dfs[ANSYS].columns[B])
    plt.legend()
    plt.grid()

    """HAM to ANSYS"""
    rangeAnsys = max(dfs[HAM].iloc[:, B]) - min(dfs[HAM].iloc[:, B])
    uShiftedAnsys = shiftPlotYdirection(dfs[ANSYS].iloc[:, B], shiftPercent*rangeAnsys)
    lShiftedAnsys = shiftPlotYdirection(dfs[ANSYS].iloc[:, B], -shiftPercent*rangeAnsys)

    fig1, ax1 = plt.subplots()
    # Shade in the area between the lines
    ax1.fill_between(x_series, uShiftedAnsys, lShiftedAnsys, where=uShiftedAnsys >= lShiftedAnsys, interpolate=True, alpha=0.25, color='green')
    # Plot the actual data
    ax1.plot(x_series, yANSYS_series, '-', color="green", lw=1, ms=2, label="ANSYS")
    ax1.plot(x_series, yHAM_series, 'x', color="#003f5c", lw=1, ms=2, label="HAM")

    digits = 3
    meanErrAnsysHam = round(errorOfCurves(dfs, ANSYS, HAM), digits)
    biasErrAnsysHam = percentPointsBetweenCurves(lShiftedAnsys, uShiftedAnsys, yHAM_series)
    boxStr = f'Mean Absolute Error:\n' \
             f'{ANSYS}-{HAM}: {meanErrAnsysHam}\n' \
             f'Bias Error:\n' \
             f'{ANSYS}-{HAM}: {round(biasErrAnsysHam, 2)}%\n'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax1.text(0.05, 0.95, boxStr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.title(type_)
    plt.xlabel(dfs[ANSYS].columns[POS])
    plt.ylabel(dfs[ANSYS].columns[B])
    plt.legend()
    plt.grid()
    # Show the plot
    plt.show()

    # Plot the histograms
    plt.hist(yBASE_series-yHAM_series, bins=60, density=True, alpha=0.5, color='blue', label="HAM")
    plt.hist(yANSYS_series-yHAM_series, bins=60, density=True, alpha=0.5, color='#003f5c', label="ANSYS")
    # Add labels and title
    plt.title(type_)
    plt.xlabel('Error [T]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


def shiftPlotYdirection(y0, bias):
    return np.array([x + bias for x in y0])


def percentPointsBetweenCurves(lBound, uBound, measured):
    return len([measured[x] for x in range(len(measured)) if lBound[x] < measured[x] < uBound[x]]) / len(measured) * 100


def errorOfCurves(dfs, key1, key2):
    return float(np.mean(np.abs(np.array(dfs[key1].iloc[:, B]) - np.array(dfs[key2].iloc[:, B]))))


def plotForceFromFile(df_file):
    df = pd.read_csv(df_file)
    # Find the index of the last '\' character
    last_slash_index = df_file.rfind('\\')
    # Use string slicing to extract the string after the last '\' character
    motorFromFileName = df_file[last_slash_index + 1:].replace(".csv", "").split("_")
    slots = int(motorFromFileName[0])
    pole_pairs = int(motorFromFileName[1])
    motorCfg = {"slots": slots, "pole_pairs": pole_pairs, "length": 0.27, "windingLayers": 2, "windingShift": "auto"}
    motor = LimMotor(motorCfg, buildBaseline=False)
    mass = motor.massTot

    TIME = 0
    FX = 1
    FY = 2
    newLen = -len(df.iloc[:, FX]) // 5
    fxArray = df.iloc[newLen:, FX]
    fyArray = df.iloc[newLen:, FY]
    timeArray = df.iloc[newLen:, TIME]

    motor.Fx = round(sum(fxArray)/len(fxArray), 2)
    motor.Fy = round(sum(fyArray)/len(fyArray), 2)

    plt.plot(df.iloc[:, TIME], df.iloc[:, FX],
             label=f'{df.columns[FX]} = {motor.Fx} N @ (Ns={slots}, Np={2*pole_pairs})')
    plt.plot(df.iloc[:, TIME], df.iloc[:, FY],
             label=f'{df.columns[FY]} = {motor.Fy} N @ (Ns={slots}, Np={2*pole_pairs})')

    plt.xlabel(df.columns[TIME])
    plt.ylabel("Force [N]")
    plt.legend(loc="center right", title=f"Steady state average [{timeArray.iloc[0]}, {timeArray.iloc[-1]}] ms")
    plt.grid()

    return plt, motor


def plotMotorThrusts():
    forceDataFolder = os.path.join(CWD, 'ForceData')
    plot = None
    motors = []
    for fileName in os.listdir(forceDataFolder):
        file = os.path.join(forceDataFolder, fileName)
        if os.path.isfile(file):
            plot, motor = plotForceFromFile(file)
            motors.append(motor)
    if plot is not None:
        plot.show()

    pareto_plot = []
    phases = 3
    q_values = list({m.slots/(2 * m.polePairs)/phases for m in motors})
    cmap = plt.get_cmap('plasma')
    color_map = [colors.to_hex(cmap(i)) for i in range(0, len(cmap.colors), len(cmap.colors)//len(q_values))][:len(q_values)]
    track_q_cmap = copy.deepcopy(color_map)
    copyCmap1 = copy.deepcopy(track_q_cmap)
    copyCmap2 = copy.deepcopy(track_q_cmap)
    fig1, ax1 = plt.subplots()
    for cnt, m in enumerate(motors):
        Fx, Fy = m.Fx, m.Fy
        slots, poles = m.slots, 2 * m.polePairs
        q = slots/poles/phases
        color_index = q_values.index(q)
        color = color_map[color_index]
        if color in copyCmap1:
            ax1.scatter(q, Fx, color=color, marker='x', label=f"Fx, q={q}")
            ax1.scatter(q, Fy, color=color, marker='o', label=f"Fy, q={q}")
            copyCmap1.remove(color)
        else:
            ax1.scatter(q, Fx, color=color, marker='x')
            ax1.scatter(q, Fy, color=color, marker='o')
        ax1.text(q, Fx, f" ({slots},{poles})", ha='left', va='bottom', fontsize="small")
        ax1.text(q, Fy, f" ({slots},{poles})", ha='left', va='bottom', fontsize="small")

    plt.xticks(q_values, q_values)
    plt.xlabel("q [-]")
    plt.ylabel("Force [N]")
    ax1.legend(loc="center right")
    plt.show()

    # Plot thrust and mass per motor
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    for cnt, m in enumerate(motors):
        q = m.slots/(2 * m.polePairs)/phases
        color_index = q_values.index(q)
        color = color_map[color_index]
        if color in copyCmap2:
            ax2.scatter(m.massTot, m.Fx, color=color, marker='o', label=f"q={q}")
            copyCmap2.remove(color)
        else:
            ax2.scatter(m.massTot, m.Fx, color=color, marker='o')
        ax2.text(m.massTot, m.Fx, f" ({m.slots},{2 * m.polePairs})", ha='left', va='bottom', fontsize="small")

    plt.xlabel("mass [kg]")
    plt.ylabel("Fx [N]")
    ax2.legend(loc="center right")
    plt.show()


plotAirgapBfield('Bx')
plotAirgapBfield('By')
# plotMotorThrusts()
