from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
from platypus import *

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
https://www.indusmic.com/post/schwefel-function
'''

PROJECT_PATH = os.path.abspath(os.path.join(__file__, "../.."))
OUTPUT_PATH = os.path.join(PROJECT_PATH, 'Optimizations')


# Problem is the object that Test will inherit from
class Constraints(Problem):

    def __init__(self, _lower, _upper):
        # The numbers indicate: #inputs, #objectives, #constraints
        super(Constraints, self).__init__(2, 1)
        # Constrain the range and type for each input
        self.types[:] = [Real(_lower, _upper), Real(_lower, _upper)]
        # Choose which objective to maximize and minimize
        self.directions[:] = [self.MINIMIZE]

    def evaluate(self, solution):
        x = solution.variables[0]
        y = solution.variables[1]
        magnitude_fitness = schwefel(x, y)
        solution.objectives[:] = [magnitude_fitness]


def plotResults(str_name, algorithm):
    if not algorithm:
        return
    feasible_solutions = [s for s in unique(nondominated(algorithm.result))]
    plotResults = []
    for results in feasible_solutions:
        x, y = (algorithm.problem.types[0].decode(results.variables[0]), algorithm.problem.types[0].decode(results.variables[1]))
        objective = results.objectives[0]
        print(f'coords: {(x, y)} objective: {objective}')
        plotResults.append((x, y, objective))

    fig, axs = plt.subplots(3)
    fig.suptitle(str_name)
    axs[0].scatter(range(len(feasible_solutions)), [x[0] for x in plotResults])
    axs[0].set(ylabel='x')
    axs[1].scatter(range(len(feasible_solutions)), [x[1] for x in plotResults])
    axs[1].set(ylabel='y')
    axs[2].scatter(range(len(feasible_solutions)), [x[2] for x in plotResults])
    axs[2].set(ylabel='objective')
    plt.xlabel('Generations')
    plt.show()


def nsgaii(_lower, _upper, run=False):
    if not run:
        return
    function_evals = 5000
    population_size = 500
    algorithm = NSGAII(Constraints(_lower, _upper), population_size)
    algorithm.run(function_evals)
    return algorithm


def ga(_lower, _upper, run=False):
    if not run:
        return
    function_evals = 5000
    population_size = 500
    offspring_size = 100
    algorithm = GeneticAlgorithm(Constraints(_lower, _upper), population_size, offspring_size)
    algorithm.run(function_evals)
    return algorithm


def pso(_lower, _upper, run=False):
    if not run:
        return

    function_evals = 5000
    swarm_size = 500
    leader_size = 100

    # https: // deap.readthedocs.io / en / master / api / tools.html  # deap.tools.mutGaussian
    algorithm = ParticleSwarm(Constraints(_lower, _upper), swarm_size=swarm_size, leader_size=leader_size,
                              generator=RandomGenerator(), leader_comparator=AttributeDominance(crowding_distance_key),
                              dominance=ParetoDominance(), fitness=crowding_distance, fitness_getter=crowding_distance_key)
    algorithm.run(function_evals)
    return algorithm


def schwefel(x1,x2):
    dims = 2
    return 418.9829 * dims - x1 * np.sin(np.sqrt(abs(x1))) - x2 * np.sin(np.sqrt(abs(x2)))


def profile_main():

    import cProfile, pstats, io

    lower, upper, num = -500, 500, 100
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)

    prof = cProfile.Profile()
    prof = prof.runctx("nsgaii(lower, upper, run=True)", globals(), locals())

    stream = io.StringIO()

    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("time")  # or cumulative
    stats.print_stats(80)  # 80 = how many to print

    # The rest is optional.
    # stats.print_callees()
    # stats.print_callers()

    # logging.info("Profile data:\n%s", stream.getvalue())

    f = open(os.path.join(OUTPUT_PATH, 'profile.txt'), 'a')
    f.write(stream.getvalue())
    f.close()


def main():
    lower, upper, num = -500, 500, 100
    x1 = np.linspace(lower, upper, num)
    x2 = np.linspace(lower, upper, num)

    # NSGAII
    plotResults('NSGAII', nsgaii(lower, upper, run=False))

    # GA
    plotResults('GA', ga(lower, upper, run=False))

    # PSO
    plotResults('PSO', pso(lower, upper, run=True))

    # Plotting
    x1, x2 = np.meshgrid(x1, x2)
    results = schwefel(x1, x2)
    figure = plt.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x1, x2, results, cmap=cm.jet, linewidth=0, antialiased=False)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    plt.show()


if __name__ == '__main__':
    # profile_main()
    main()
