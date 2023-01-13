from Optimizations.OptimizationCfg import *
from platypus import *

'''
https://platypus.readthedocs.io/en/latest/getting-started.html
'''


def platypus(motorCfg, hamCfg, canvasCfg, run=False):
    '''
    This is a function that iterates over many motor configurations while optimizing for performance
    '''

    if not run:
        return

    logging.FileHandler(LOGGER_FILE, mode='w')

    tolerance = 10 ** (-7)
    max_evals = 30000 - 1
    max_stalls = 25
    stall_tolerance = tolerance
    timeout = 3000000  # seconds
    parent_size = 200
    tournament_size = 2
    constraint_params = {'slots': [1, motorCfg['slots']], 'pole_pairs': [1, motorCfg['pole_pairs']],
                         'motorCfg': motorCfg, 'hamCfg': hamCfg, 'canvasCfg': canvasCfg}
    termination_params = {'max_evals': max_evals, 'tolerance': tolerance,
                          'max_stalls': max_stalls, 'stall_tolerance': stall_tolerance,
                          'timeout': timeout}
    solverList = []
    nsga_params = {'population_size': parent_size, 'generator': RandomGenerator(),
                   'selector': TournamentSelector(tournament_size), 'variator': GAOperator(SBX(0.3), PM(0.1)),
                   'archive': FitnessArchive(nondominated_sort)}
    solverList = solveOptimization(NSGAII, MotorOptProblem, solverList, constraint_params, termination_params, nsga_params, run=True)


def profile_main():

    import cProfile, pstats, io

    prof = cProfile.Profile()
    prof = prof.runctx("main()", globals(), locals())

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

    # ___Baseline motor configurations___
    buildMotor(run=False, baseline=True,
               motorCfg={"slots": 16, "pole_pairs": 3, "length": 0.27, "windingLayers": 2, "windingShift": 2},
               # If invertY == False -> [LowerSlot, UpperSlot, Yoke]
               hamCfg={"N": 100, "errorTolerance": 1e-15, "invertY": False,
                       "hmRegions": {1: "vac_lower", 2: "bi", 3: "dr", 4: "g", 6: "vac_upper"},
                       "mecRegions": {5: "mec"}},
               canvasCfg={"pixDiv": [15, 15], "canvasSpacing": 80, "fieldType": "B",
                          "showAirGapPlot": False, "showUnknowns": False, "showGrid": True, "showFields": True,
                          "showFilter": True, "showMatrix": False, "showZeros": True})

    # ___Custom Configuration___
    motorCfg = {"slots": 54, "pole_pairs": 3, "length": 0.27, "windingLayers": 2, "windingShift": "auto"}
    hamCfg = {"N": 100, "errorTolerance": 1e-15, "invertY": False,
              "hmRegions": {1: "vac_lower", 2: "bi", 3: "dr", 4: "g", 6: "vac_upper"},
              "mecRegions": {5: "mec"}}
    canvasCfg = {"pixDiv": [5, 5], "canvasSpacing": 80, "fieldType": "B",
                 "showAirGapPlot": False, "showUnknowns": False, "showGrid": True, "showFields": True,
                 "showFilter": True, "showMatrix": False, "showZeros": True}

    # ___Motor optimization___
    platypus(run=False, motorCfg=motorCfg, hamCfg=hamCfg, canvasCfg=canvasCfg)

    # ___Custom motor model___
    buildMotor(run=True, motorCfg=motorCfg, hamCfg=hamCfg, canvasCfg=canvasCfg)

    slots, pole_pairs = [12, 55], [2, 8]
    p = MotorOptProblem(slots, pole_pairs, motorCfg, hamCfg, canvasCfg)
    motorList = p.uniqueCombinations()
    print("All possible motors: ", motorList)


if __name__ == '__main__':
    # profile_main()  # To profile the main execution
    main()
