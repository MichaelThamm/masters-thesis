"""
Implementation of CUDA accelerated random walk pagerank.

The implementation is optimized with the CUDA JIT in numba
for single GPU execution.
"""
import numpy as np

from numba import uint64, uint32, float32, cuda
from pyculib import sorting


MAX32 = uint32(0xffffffff)


@cuda.jit("(uint64[::1], uint64)", device=True)
def cuda_xorshift(states, id):
    x = states[id]
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    states[id] = x
    return uint64(x) * uint64(2685821657736338717)


@cuda.jit("float32(uint64[::1], uint64)", device=True)
def cuda_xorshift_float(states, id):
    return float32(float32(MAX32 & cuda_xorshift(states, id)) / float32(MAX32))


@cuda.jit(device=True)
def cuda_random_walk_per_node(curnode, visits, colidx, edges, resetprob,
                              randstates):
    tid = cuda.threadIdx.x
    randnum = cuda_xorshift_float(randstates, tid)
    if randnum >= resetprob:
        base = colidx[curnode]
        offset = colidx[curnode + 1]
        # If the edge list is non-empty
        if offset - base > 0:
            # Pick a random destination
            randint = cuda_xorshift(randstates, tid)
            randdestid = (uint64(randint % uint64(offset - base)) +
                          uint64(base))
            dest = edges[randdestid]
        else:
            # Random restart
            randint = cuda_xorshift(randstates, tid)
            randdestid = randint % uint64(visits.size)
            dest = randdestid

        # Increment visit count
        cuda.atomic.add(visits, dest, 1)


MAX_TPB = 64 * 2


@cuda.jit("void(uint32[::1], uint32[::1], uint32[::1], uint32[::1], float32, "
          "uint64[::1], uint32[::1])")
def cuda_random_walk_round(coupons, visits, colidx, edges, resetprob,
                           randstates, remap):
    sm_randstates = cuda.shared.array(MAX_TPB, dtype=uint64)

    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x

    if blkid < coupons.size:
        workitem = remap[blkid]
        sm_randstates[tid] = randstates[workitem] + tid
        count = coupons[workitem]

        # While there are coupons
        while count:
            # Try to assign coupons to every thread in the block
            assigned = min(count, cuda.blockDim.x)
            count -= assigned
            # Thread within assigned range
            if tid < assigned:
                cuda_random_walk_per_node(workitem, visits, colidx, edges,
                                          resetprob, sm_randstates)
            # Kill the thread
            else:
                return

        if tid == 0:
            randstates[workitem] = sm_randstates[tid]


@cuda.jit("void(uint32[::1], uint32[::1])")
def cuda_reset_and_add_visits(visits, total):
    tid = cuda.grid(1)
    if tid < visits.size:
        total[tid] += visits[tid]
        visits[tid] = 0


@cuda.jit("void(uint32[::1], uint32[::1])")
def cuda_search_non_zero(bins, count):
    loc = cuda.grid(1)
    if loc < bins.size:
        if bins[loc] != 0:
            cuda.atomic.add(count, 0, 1)


def random_walk(nodes, coupons, colidx, edges, resetprob):
    visits = np.zeros(len(nodes), dtype=np.uint32)
    numnodes = len(nodes)

    randstates = np.random.randint(1, 0xffffffff, size=len(nodes)).astype(
        np.uint64)

    d_remap = cuda.to_device(np.arange(len(nodes), dtype=np.uint32))

    nnz = np.array([len(nodes)], dtype=np.uint32)
    d_nnz = cuda.to_device(nnz)
    d_coupons = cuda.to_device(coupons)
    d_visits = cuda.to_device(visits)
    d_visits_tmp = cuda.to_device(visits)

    d_colidx = cuda.to_device(colidx)
    d_edges = cuda.to_device(edges)
    d_randstates = cuda.to_device(randstates)

    d_total = cuda.to_device(visits)

    round_count = 0

    sorter = sorting.RadixSort(maxcount=d_remap.size, dtype=d_visits.dtype,
                               descending=True)

    while nnz[0]:  # and round_count < 3:
        round_count += 1
        # Run random walk kernel
        device_args = (d_coupons, d_visits, d_colidx, d_edges, resetprob,
                       d_randstates, d_remap)

        cuda_random_walk_round[nnz[0], MAX_TPB](*device_args)

        # Prepare for next round
        d_coupons.copy_to_device(d_visits)
        d_visits_tmp.copy_to_device(d_visits)

        # Remap indices to that earlier ones have more to do

        d_remap = sorter.argsort(keys=d_visits_tmp)

        d_nnz.copy_to_device(np.zeros(1, dtype=np.uint32))
        cuda_search_non_zero.forall(d_visits_tmp.size)(d_visits_tmp,
                                                       d_nnz)
        nnz = d_nnz.copy_to_host()

        cuda_reset_and_add_visits.forall(numnodes)(d_visits, d_total)

    return d_total.copy_to_host()

