import time
from typing import List, Callable, Tuple, Dict

from Simulation import total_loss, calc_wait_time
from System import System


def compare_algorithms(system: System,
                 algorithms: List[Callable],
                 lim_seconds: int = 9223372036854775807) -> \
        Tuple[List[Tuple[Dict[int, List[Tuple[int, int]]],
                         Dict[int, List[Tuple[int, int]]]]],
              List[List[int]]]:
    """
    compare list of optimize algorithms
    :param system: the system we optimize
    :param algorithms: list of optimize algorithms to check
    :param lim_seconds: limit time for the runtime
    :return: best solution of each algorithm and best fitness of each iteration in the algorithm
    """
    # each algo returns (best final sol,best fitness of each iteration)
    best_sols = []
    bests_each_iteration = []
    for i, algo in enumerate(algorithms):
        start = time.time()
        (best_sol, best_each_iteration) = algo(system, lim_seconds=lim_seconds)
        algo1_time = time.time() - start
        print(f"Algo {i} Loss After {algo1_time} Seconds:")
        print(total_loss(system, best_sol[0], get_hint=False))
        best_sols.append(best_sol)
        bests_each_iteration.append(best_each_iteration.copy())
        # calc_wait_time.cache_clear()
        # total_loss.cache_clear()

    return best_sols, bests_each_iteration


def more_information(system: System, sol: Tuple[Dict[int, List[Tuple[int, int]]],
                                                Dict[int, List[Tuple[int, int]]]]) \
        -> Tuple[int, int, Dict[int, int], int, float]:
    """
    get the following information about the solution:

    - total time waited

    - total number of people that doesn't got on a bus

    - number of people that doesn't got on a bus in each line

    - the total punishment in the wait time for people that doesn't got on a bus

    - average time waited (for people who got on a bus)

    :param system: the system we work in
    :param sol: the solution to get data for
    :return: tha data mentioned above
    """
    time_waited, not_picked, punishment = calc_wait_time(system, sol[0], more_info=True)
    return (time_waited, sum(not_picked.values()), not_picked, punishment,
            (time_waited - punishment) / (len(system.passengers()) - sum(not_picked.values())))
