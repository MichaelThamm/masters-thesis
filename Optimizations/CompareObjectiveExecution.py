import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def schwefelWait(x1, x2):
    _start = perf_counter()
    _stop = _start
    while _stop - _start < 0.05:
        _stop = perf_counter()
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


def avgTimeSchwefel(count, run=False):
    if not run:
        return

    times = []
    for i in range(count):
        start = perf_counter()
        schwefelWait(i * np.random.uniform(-500.0, 500.0), i * np.random.uniform(-500.0, 500.0))
        stop = perf_counter()
        times.append(stop - start)
    print(f'with sleep: {sum(times) / len(times)}, {times}')


def plotTimeComparison(run=False):
    if not run:
        return

    timeSlices = [0.0086, 1.0114, 10.0168, 50.0217, 100.0208]  # ms
    gaNfe = [4170.6, 5762.6, 4248.4, 5281, 4257.2]
    gaTime = [1.4545994, 7.499245, 43.785079, 265.742423, 427.649698]  # s
    psoNfe = [10760, 13400, 13280, 11880, 12520]
    psoTime = [1.5245998, 15.0024522, 138.2456215, 595.657676, 1254.964626]  # s
    new = timeSlices + list(map(lambda _tuple: (_tuple[1] + timeSlices[_tuple[0] + 1])/2, enumerate(timeSlices[:-1])))
    new.sort()

    f_gaTime = interp1d(timeSlices, gaTime, kind='cubic')
    f_psoTime = interp1d(timeSlices, psoTime, kind='cubic')
    plt.plot(new, f_gaTime(new), 'o-', color='orange')
    plt.plot(new, f_psoTime(new), 'o-', color='purple')
    plt.legend(['GA Time', 'PSO Time'], loc='best')
    plt.xlabel('Objective Function Execution Time (ms)')
    plt.ylabel('Solver Execution Time (s)')
    plt.title('Solver Execution Time vs\nObjective Function Execution Time')
    plt.show()


avgTimeSchwefel(100, run=False)
plotTimeComparison(run=True)

# # x = np.linspace(0, 10, num=11, endpoint=True)
# x = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
# y = [1, 0.99383351, 0.90284967, 0.54030231, -0.20550672, -0.93454613, -0.65364362, 0.6683999, 0.67640492, -0.91113026, 0.11527995]
# f = interp1d(x, y)
# f2 = interp1d(x, y, kind='cubic')
# xnew = np.linspace(0, 10, num=41, endpoint=True)
# plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
# plt.legend(['data', 'linear', 'cubic'], loc='best')
# plt.show()