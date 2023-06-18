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
    gaTime = [1.4545994, 7.499245, 43.785079, 265.742423, 427.649698]  # s
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