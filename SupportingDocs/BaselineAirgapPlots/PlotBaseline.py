import copy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import scipy

random.seed(10)

ANSYS = 'Ansys'
BASE = 'Baseline'
HAM = 'HAM'
POS = 0
B = 1


def plotAirgap(type_):

    ansysFile = f'{ANSYS+type_}.csv'
    baseFile = f'{BASE+type_}.csv'
    dfs = {ANSYS: pd.read_csv(ansysFile, usecols=["Distance [mm]", f'{type_} [T]']),
           BASE: pd.read_csv(baseFile)}

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
    dfs[BASE] = scipy.interpolate.interp1d(dfs[BASE].iloc[:, POS], dfs[BASE].iloc[:, B], "linear")
    dfs[HAM] = scipy.interpolate.interp1d(dfs[HAM].iloc[:, POS], dfs[HAM].iloc[:, B], "linear")
    # Turn interpolation back into DataFrame
    dfs[BASE] = pd.DataFrame({dfs[ANSYS].columns[POS]: dfs[ANSYS].iloc[:, POS],
                              dfs[ANSYS].columns[B]: dfs[BASE](dfs[ANSYS].iloc[:, POS])})
    dfs[HAM] = pd.DataFrame({dfs[ANSYS].columns[POS]: dfs[ANSYS].iloc[:, POS],
                             dfs[ANSYS].columns[B]: dfs[HAM](dfs[ANSYS].iloc[:, POS])})

    # Plot airgap
    fig, ax = plt.subplots()
    for key, data in dfs.items():
        if key == HAM:
            ax.plot(data.iloc[:, POS], data.iloc[:, B], 'xb', lw=1, ms=2, label=key)
        elif key == BASE:
            ax.plot(data.iloc[:, POS], data.iloc[:, B], '-r', label=key)
        else:
            ax.plot(data.iloc[:, POS], data.iloc[:, B], label=key)

    digs = 3
    errAnsysBase = round(errorOfCurves(dfs, ANSYS, BASE), digs)
    errAnsysHam = round(errorOfCurves(dfs, ANSYS, HAM), digs)
    errBaseHam = round(errorOfCurves(dfs, BASE, HAM), digs)

    boxStr = f'Mean Error Between Curves:\n' \
             f'{ANSYS}-{BASE}: {errAnsysBase}\n' \
             f'{ANSYS}-{HAM}: {errAnsysHam}\n' \
             f'{BASE}-{HAM}: {errBaseHam}'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, boxStr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.title("")
    plt.xlabel(dfs[ANSYS].columns[POS])
    plt.ylabel(dfs[ANSYS].columns[B])
    plt.legend()
    plt.grid()
    plt.show()

    print('Ansys to Base: ', errAnsysBase,
          'Ansys to HAM: ', errAnsysHam,
          'Base to HAM: ', errBaseHam)


def errorOfCurves(dfs, key1, key2):
    return float(np.mean(np.abs(np.array(dfs[key1].iloc[:, B]) - np.array(dfs[key2].iloc[:, B]))))


def plotForce(df_file):
    df = pd.read_csv(df_file)
    TIME = 0
    FX = 1
    FY = 2
    plt.plot(df.iloc[:, TIME], df.iloc[:, FX], label=f'{df.columns[FX]}')
    plt.plot(df.iloc[:, TIME], df.iloc[:, FY], label=f'{df.columns[FY]}')

    plt.xlabel(df.columns[TIME])
    plt.ylabel("Force [N]")
    plt.legend()
    plt.grid()
    plt.show()


plotAirgap('Bx')
plotAirgap('By')
plotForce('ForcePlot.csv')