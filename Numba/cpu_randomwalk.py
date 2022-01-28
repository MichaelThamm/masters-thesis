"""
Implementation of CPU random walk pagerank.

The implementation is optimized with the numba JIT compiler
and uses multi-threads for parallel execution.
"""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import os

from numba import jit, float32, uint64, uint32
import numpy as np


CPU_COUNT = int(os.environ.get('CPU_COUNT', multiprocessing.cpu_count()))

MAX32 = uint32(0xffffffff)


@jit("(uint64[::1], uint64)", nogil=True)
def xorshift(states, id):
    x = states[id]
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    states[id] = x
    return uint64(x) * uint64(2685821657736338717)


@jit("float32(uint64[::1], uint64)", nogil=True)
def xorshift_float(states, id):
    return float32(float32(MAX32 & xorshift(states, id)) / float32(MAX32))


@jit("boolean(intp, uint32[::1], uint32[::1], uint32[::1], uint32[::1], "
     "float32, uint64[::1])", nogil=True)
def random_walk_per_node(curnode, coupons, visits, colidx, edges, resetprob,
                         randstates):
    moving = False
    for _ in range(coupons[curnode]):
        randnum = xorshift_float(randstates, curnode)
        if randnum < resetprob:
            # Terminate coupon
            continue
        else:
            base = colidx[curnode]
            offset = colidx[curnode + 1]
            # If the edge list is non-empty

            if offset - base > 0:
                # Pick a random destination
                randint = xorshift(randstates, curnode)
                randdestid = uint64(randint % uint64(offset - base)) + uint64(base)
                dest = edges[randdestid]
            else:
                # Random restart
                randint = xorshift(randstates, curnode)
                randdestid = randint % uint64(visits.size)
                dest = randdestid

            # Increment visit count
            visits[dest] += 1
            moving = True

    return moving


@jit(nogil=True)
def job(tid, step, coupons, visits, colidx, edges, resetprob, randstates):
    moving = False
    base = step * tid
    numnodes = colidx.size - 1
    for i in range(base, min(base + step, numnodes)):
        # XXX: numba bug with returned boolean types
        if 1 == random_walk_per_node(i, coupons, visits, colidx, edges,
                                     resetprob, randstates):
            moving = True
    return moving


@jit(nogil=True)
def sum_vertical(temp, visits, start, stop):
    # for n in range(visits.size):
    for n in range(start, stop):
        for i in range(temp.shape[0]):
            visits[n] += temp[i, n]


def random_walk_round(coupons, visits, colidx, edges, resetprob, randstates):
    npar = CPU_COUNT
    numnodes = colidx.size - 1
    split_visits = np.zeros((npar, visits.size), dtype=visits.dtype)

    assert numnodes == visits.size
    step = int(np.ceil(numnodes / npar))

    with ThreadPoolExecutor(max_workers=npar) as e:
        futures = [e.submit(job, *(tid, step, coupons, split_visits[tid],
                                   colidx, edges, resetprob, randstates))
                   for tid in range(npar)]
        moving = any(f.result() for f in futures)

    # with ThreadPoolExecutor(max_workers=npar) as e:
        futures = [
            e.submit(sum_vertical, *(split_visits, visits, n, min(n + step, visits.size)))
            for n in range(0, visits.size, step)
            ]
        [f.result() for f in futures]

    return moving


def random_walk(nodes, coupons, colidx, edges, resetprob):
    visits = np.zeros(len(nodes), dtype=np.uint32)
    total = visits.copy()

    randstates = np.random.randint(1, 0xffffffff, size=len(nodes)).astype(np.uint64)

    while True:
        visits.fill(0)
        moving = random_walk_round(coupons, visits, colidx, edges, resetprob,
                                   randstates)
        if not moving:
            break
        else:
            coupons[:] = visits[:]
            total += visits

    return total
