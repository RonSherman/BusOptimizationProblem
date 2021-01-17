import math
import random
import time
from collections import defaultdict
from typing import List, Tuple, Dict, Union

import numpy as np
import matplotlib.pyplot as plot

from GeneticAlgorithm import change_minute_random, change_size_random, union_sol_of_lines
from Simulation import total_loss
from System import System
from SystemGenerator import create_random_bus_line, create_random_bus, min_bus_map_to_line, create_population


def plot_graph_from_the_middle(name: str,
                               data: List[List[int]],
                               start_from: int = 100) -> None:
    """
    plot graph of the best fitness of each iteration from iteration i
    :param name: title
    :param data: data we plot
    :param start_from: the iteration we plot from
    """
    plot.title(name)
    for i in range(len(data)):
        fitness = data[i][start_from:]
        plot.plot(fitness, label=f'Best fitness Round {i}')
    text = plot.xticks()[1]
    text[1] = plot.Text(0, 0, str(start_from))
    text[2:] = [plot.Text(0, 0, str(i)) for i in plot.xticks()[0][2:]]
    plot.xticks(ticks=plot.xticks()[0], labels=text)
    plot.xlabel("Iteration")
    plot.ylabel("Fitness")
    plot.legend()
    plot.show()


def simulated_annealing(system: System,
                        lim_seconds: int = 9223372036854775807,
                        with_size_adviser: bool = True,
                        initial_temp: int = 90,
                        alpha: float = 0.05,
                        show_graph: bool = True) -> Tuple[Tuple[Dict[int, List[Tuple[int, int]]],
                                                                Dict[int, List[Tuple[int, int]]]],
                                                          List[List[int]]]:
    """
    runs the simulated annealing algorithm on a given bus system
    :param system: the system we optimize
    :param lim_seconds: optional limit for runtime
    :param with_size_adviser: optional using size adviser
    :param initial_temp: initial temperature
    :param alpha: decrease rate for the temperature every round
    :param show_graph: if true show graph of the best fitness throughout the running
    :return: best solution founded and best fitness of each iteration
    """

    final_temp = .1
    num_rounds = 5
    best_outputs = [[] for _ in range(num_rounds)]
    # states_fitness = []
    solution: List[Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]] \
        = [None for _ in range(num_rounds)]
    # if
    start = time.time()
    # random restart- do a couple of rounds
    for i in range(num_rounds):
        # create our initial state
        initial_state = create_population(system, start_population=1)[0]
        current_temp = initial_temp
        # Start by initializing the current state with the initial state
        current_state = initial_state
        solution[i] = current_state
        # check for size heuristic
        if with_size_adviser:
            solution[i], solution_fitness = get_cost(system, solution[i])
        else:
            solution_fitness = get_cost(system, solution[i], with_size_adviser=False)
        # run for as long as temp is bigger than stopping point, and check for time passed
        while current_temp > final_temp and time.time() - start < lim_seconds:
            # check for size heuristic
            if with_size_adviser:
                current_state, fitness = get_cost(system, current_state)
            else:
                fitness = get_cost(system, current_state, with_size_adviser=False)
            # get the neighbor using random mutations
            neighbor = get_neighbor(system, current_state, with_size_adviser)
            # check for size heuristic
            if with_size_adviser:
                neighbor, neighbor_fitness = get_cost(system, neighbor)
            else:
                neighbor_fitness = get_cost(system, neighbor, with_size_adviser=False)
            # Check if neighbor is best so far
            cost_diff = fitness - neighbor_fitness

            # if the new solution is better, accept it
            if cost_diff > 0:
                solution[i] = neighbor
                current_state = neighbor
                if neighbor_fitness < solution_fitness:
                    solution_fitness = neighbor_fitness
            # if the new solution is not better, accept it with a probability of e^(-cost/temp)
            else:
                if random.uniform(0, 1) < math.exp(cost_diff / current_temp):
                    current_state = neighbor
            # decrement the temperature
            current_temp -= alpha
            # states_fitness.append(fitness)
            best_outputs[i].append(solution_fitness)
    # print the graph
    if show_graph:
        plot_graph_from_the_middle('Simulated Annealing', best_outputs)

    best_rounds = [best_outputs[i][-1] for i in range(num_rounds) if len(best_outputs[i]) > 0]
    # maxed_val = max(best_rounds)
    print(best_rounds)

    return solution[int(np.argmin(best_rounds))], best_outputs


def get_cost(system: System,
             state: Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]],
             with_size_adviser: bool = True) -> Union[int, Tuple[Tuple[Dict[int, List[Tuple[int, int]]],
                                                                       Dict[int, List[Tuple[int, int]]]], int]]:
    """
    Calculates the cost of some state
    :param system: the system we optimize
    :param state: the state to get his cost
    :param with_size_adviser: optional use size adviser
    :return: cost of the state / cost of the improved state+his cost
    """
    if not with_size_adviser:
        return total_loss(system, state[0])
    """Calculates cost of the argument state for your solution."""
    _, sol_by_min, cost = total_loss(system, state[0], get_hint=True)
    sol = sol_by_min, min_bus_map_to_line(sol_by_min)
    return sol, cost


def get_neighbor(system: System,
                 state: Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]],
                 with_size_adviser: bool = True) -> Tuple[Dict[int, List[Tuple[int, int]]],
                                                          Dict[int, List[Tuple[int, int]]]]:
    """
    :param system: the system we optimize
    :param state: current state to get neighbour
    :param with_size_adviser: optional use size adviser
    :return: neighbors of the argument state for your solution.
    """

    bus_map_by_min, bus_map_by_line_id = state
    sizes = system.sizes()
    lines = system.lines()
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    bus_map_by_min_after = defaultdict(list)
    new_bus_map_by_line_id = defaultdict(list)
    # gene_idx = mutations_counter - 1
    for line in bus_map_by_line_id:
        # print(offspring_crossover,idx)
        cur_busses = bus_map_by_line_id[line]
        new_bus_map_by_line_id[line] = cur_busses.copy()
        new_cur_busses = new_bus_map_by_line_id[line]
        '''
        add/remove bus to the line
        '''
        r = random.random()
        remove_prob = 0.4 if with_size_adviser else 0.7
        if r < 0.35:
            minute, size = create_random_bus(system, line, sizes)
            new_cur_busses.append((minute, size))
        elif r < remove_prob and len(bus_map_by_line_id[line]) > 0:
            # print(offspring_crossover,idx,line)

            i = random.randrange(len(bus_map_by_line_id[line]))  # get random index
            # print(cur_busses,i)
            new_cur_busses[i], new_cur_busses[-1] = new_cur_busses[-1], new_cur_busses[i]  # swap with the last element
            new_cur_busses.pop()  # pop last element O(1)
        for i, (minute, size) in enumerate(new_cur_busses):
            new_size = change_size_random(sizes, size, with_size_adviser)
            new_minute = change_minute_random(minute)
            new_cur_busses[i] = (new_minute, new_size)

    r = random.random()
    if r < 0.4:
        line_num, new_line = create_random_bus_line(system, lines, sizes)
        if line_num not in bus_map_by_line_id:
            new_bus_map_by_line_id[line_num] = new_line

    for line in new_bus_map_by_line_id:
        for minute, size in new_bus_map_by_line_id[line]:
            bus_map_by_min_after[minute].append((line, size))
    return bus_map_by_min_after, new_bus_map_by_line_id


def simulated_annealing_optimize_lines_sep(system: System,
                                           lim_seconds: int = 9223372036854775807,
                                           with_size_adviser: bool = True,
                                           initial_temp: int = 90,
                                           alpha: float = 0.05,
                                           show_graph: bool = True) -> Tuple[Tuple[Dict[int, List[Tuple[int, int]]],
                                                                                   Dict[int, List[Tuple[int, int]]]],
                                                                             List[List[int]]]:
    """
    Performs simulated annealing for each line separately to find a solution
    :param system: the system we optimize
    :param lim_seconds: optional limit for runtime
    :param with_size_adviser: optional using size adviser
    :param initial_temp: initial temperature
    :param alpha: decrease rate for the temperature every round
    :param show_graph: if true show graph of the best fitness throughout the running
    :return: best solution founded and best fitness of each iteration
    """
    best_sol = {}
    num_rounds = 5
    best_outputs = [[] for _ in range(num_rounds)]
    num_of_lines = len(system.lines())
    for line_id in system.lines():
        print(f'Start optimize line {line_id}')
        system_for_line = system.get_system_of_line(line_id)
        (best_sol[line_id], best_outputs_round) = simulated_annealing(system_for_line,
                                                                      int(lim_seconds / num_of_lines),
                                                                      with_size_adviser, initial_temp=initial_temp,
                                                                      alpha=alpha, show_graph=False)
        for i in range(num_rounds):
            if not best_outputs[i]:
                best_outputs[i] = best_outputs_round[i]
            else:
                size = min(len(best_outputs[i]), len(best_outputs_round[i]))
                temp = []
                for j in range(size):
                    temp.append(best_outputs[i][j] + best_outputs_round[i][j])
                best_outputs[i] = temp
    if show_graph:
        plot_graph_from_the_middle(f"Simulated Annealing- Lines Heuristic", best_outputs)

    return union_sol_of_lines(best_sol), best_outputs
