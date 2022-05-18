import numpy as np
import time, threading


def schwefelNone(x1, x2):
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


def schwefelWait(x1, x2):
    time.sleep(0.001)
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


times = []
for i in range(100):
    start_time = time.time()
    schwefelNone(i*np.random.uniform(-500.0, 500.0), i*np.random.uniform(-500.0, 500.0))
    endtime = time.time() - start_time
    times.append(endtime)
print(f'without sleep: {times}')

times = []
start_time = time.time()
for i in range(100):
    start_time = time.time()
    schwefelWait(i*np.random.uniform(-500.0, 500.0), i*np.random.uniform(-500.0, 500.0))
    endtime = time.time() - start_time
    times.append(endtime)
print(f'with sleep: {times}')