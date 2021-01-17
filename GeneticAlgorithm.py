import random
from collections import defaultdict
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plot
import time

from Simulation import total_loss
from System import System
from SystemGenerator import min_bus_map_to_line, create_random_bus, create_random_bus_line, create_population


def select_mating_pool_without_size_adviser(system: System,
                                            pop: List[Tuple[
                                                Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]],
                                            num_parents: int) -> List[Tuple[Dict, Dict]]:
    """
    Selecting the best individuals in the current generation as parents for producing the offspring of the next
    generation.
    :param system: the system we work in
    :param pop: the population to choose the parents from
    :param num_parents: num of parents
    :return: the parents
    """

    fitness_values = []
    for off in pop:
        fitness_values.append(total_loss(system, off[0]))
    max_ind = np.argpartition(fitness_values, num_parents)[:num_parents]
    # print(max_ind)
    # print(fitness_values)
    parents: List[Tuple[Dict, Dict]] = []
    for index in max_ind:
        parents.append(pop[index])
    return parents


def select_mating_pool(system: System,
                       pop: List[Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]],
                       num_parents: int,
                       with_size_adviser: bool = True) -> \
        List[Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]]:
    """
    Selecting the best individuals in the current generation as parents for producing the offspring of the next
    generation.
    :param with_size_adviser: if True, we use size adviser to improve the solutions
    :param system: the system we work in
    :param pop: the population to choose the parents from
    :param num_parents: num of parents
    :return: the parents
    """
    if not with_size_adviser:
        return select_mating_pool_without_size_adviser(system, pop, num_parents)
    fitness_values = []
    new_solutions = []
    for off in pop:
        _, new_sol, new_sol_fit = total_loss(system, off[0], True)
        fitness_values.append(new_sol_fit)
        new_solutions.append(new_sol)
    max_ind = np.argpartition(fitness_values, num_parents)[:num_parents]
    parents = []
    for index in max_ind:
        parents.append((new_solutions[index], min_bus_map_to_line(new_solutions[index])))
    return parents


def crossover(parents_dict_by_minute: List[Dict[int, List[Tuple[int, int]]]],
              parents_dict_by_line_id: List[Dict[int, List[Tuple[int, int]]]],
              offspring_size: int) -> Tuple[List[Dict[int, List[Tuple[int, int]]]],
                                            List[Dict[int, List[Tuple[int, int]]]]]:
    """
    crossover parents to create new offspring
    :param parents_dict_by_minute: bus map minute->list of line, size
    :param parents_dict_by_line_id: bus map line->list of minute, size
    :param offspring_size: number of offspring to create
    :return: the new offspring (bus maps by minute and by line)
    """
    # parents[par_id] : busMap : min -> list of buses (line,size)

    offspring_by_minute: List[Any] = [None] * offspring_size
    offspring_by_line_id: List[Any] = [None] * offspring_size
    # # The point at which crossover takes place between two parents. Usually, it is at the center.
    # crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size):
        parent1_idx, parent2_idx = random.sample(list(range(len(parents_dict_by_minute))), 2)

        offspring_by_minute[k] = defaultdict(list)
        child = defaultdict(list)
        busline_map1 = parents_dict_by_line_id[parent1_idx]
        busline_map2 = parents_dict_by_line_id[parent2_idx]
        for line in busline_map1:
            busses = busline_map1[line]
            child[line] = random.sample(busses, (len(busses) + 1) // 2)
        for line in busline_map2:
            busses = busline_map2[line]
            busses_to_add = random.sample(busses, (len(busses) + 1) // 2)
            for minute, size in busses_to_add:
                bus_fre = set([x[0] for x in child[line]])
                if minute not in bus_fre:
                    child[line].append((minute, size))
                elif random.random() < 0.05:
                    child[line].append((minute, size))
        offspring_by_line_id[k] = child
        for line in child:
            busses = child[line]
            for minute, size in busses:
                offspring_by_minute[k][minute].append((line, size))
    return offspring_by_minute, offspring_by_line_id


def change_size_random(sizes: List[int],
                       cur_size: int,
                       gen_number: int = 0,
                       with_size_adviser: bool = True) -> int:
    """
    change the size of the bus randomly
    :param sizes: possible sizes
    :param cur_size: current size (more chance to remain this size)
    :param gen_number: the number of generation
    :param with_size_adviser: if true can only increase the bus size
    :return: the new size
    """
    if not with_size_adviser:
        return change_size_random_without_size_adviser(sizes, cur_size, gen_number=0)
    r = random.random()
    new_sizes = [new for new in sizes if new > cur_size]
    if r < 0.02 * len(new_sizes):
        return random.choice(new_sizes)
    return cur_size


def change_size_random_without_size_adviser(sizes: List[int],
                                            cur_size: int,
                                            gen_number: int = 0) -> int:
    """
    change the size of the bus randomly
    :param sizes: possible sizes
    :param cur_size: current size (more chance to remain this size)
    :param gen_number: the number of generation
    :return: the new size
    """
    r = random.random()
    new_sizes = [new for new in sizes if new != cur_size]
    if r < 0.05:
        return random.choice(new_sizes)
    return cur_size


def change_minute_random(cur_minute: int,
                         gen_number: int = 0,
                         lower_bound: int = 0,
                         upper_bound: int = 60 * 24 - 1) -> int:
    """
    return new exit time (normally distribution around current exit time)
    :param cur_minute: current exit time (in minutes)
    :param gen_number: the number of generation
    :param lower_bound: minimum exit time
    :param upper_bound: maximum exit time
    :return: the new exit time
    """
    r = random.random()
    if r < 0.2:
        new_min = int(random.normalvariate(cur_minute, 60))
        return min(upper_bound, max(lower_bound, new_min))
    return cur_minute


def mutation(system: System,
             offspring_after_crossover_by_minute: List[Dict[int, List[Tuple[int, int]]]],
             offspring_after_crossover_by_line_id: List[Dict[int, List[Tuple[int, int]]]],
             num_mutations: int = 1, gen_number: int = 0,
             with_size_adviser=True) -> Tuple[List[Dict[int, List[Tuple[int, int]]]],
                                              List[Dict[int, List[Tuple[int, int]]]]]:
    """
    this function create mutations for the offspring come from the crossover
    just like Leonardo, Raphael, Michelangelo and Donatello
    :param system:the system we work in
    :param offspring_after_crossover_by_minute: list of offspring come from the crossover (dicts by minute)
    :param offspring_after_crossover_by_line_id: list of offspring come from the crossover (dicts by line)
    :param num_mutations: number of mutations to apply on each child
    :param gen_number: the number of the generation
    :param with_size_adviser: if True, bus sizes can only increased in the mutations
    :return: the offspring after mutations
    """
    sizes = system.sizes()
    lines = system.lines()

    # mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    offspring_after_mutation_by_minute: List[Any] = [None] * (len(offspring_after_crossover_by_minute))
    offspring_after_mutation_by_line_id: List[Any] = [None] * (len(offspring_after_crossover_by_minute))
    for idx in range(len(offspring_after_crossover_by_minute)):

        # gene_idx = mutations_counter - 1
        # for line in offspring_buslines_dict.values():
        for line in offspring_after_crossover_by_line_id[idx]:
            cur_busses = offspring_after_crossover_by_line_id[idx][line]
            for mutation_num in range(num_mutations):
                '''
                add/remove bus to the line
                '''
                r = random.random()
                remove_prob = 0.175 if with_size_adviser else 0.3
                if r < 0.15:
                    minute, size = create_random_bus(system, line, sizes)
                    cur_busses.append((minute, size))
                elif r < remove_prob and len(offspring_after_crossover_by_line_id[idx][line]) > 0:
                    # print(offspring_crossover,idx,line)

                    i = random.randrange(len(offspring_after_crossover_by_line_id[idx][line]))  # get random index
                    # print(cur_busses,i)
                    cur_busses[i], cur_busses[-1] = cur_busses[-1], cur_busses[i]  # swap with the last element
                    cur_busses.pop()  # pop last element O(1)
                for i, (minute, size) in enumerate(cur_busses):
                    new_size = change_size_random(sizes, size, gen_number, with_size_adviser)
                    new_minute = change_minute_random(minute, gen_number)
                    cur_busses[i] = (new_minute, new_size)
            # print(offspring_after_mutation_by_line_id,idx)
            offspring_after_mutation_by_line_id[idx] = defaultdict(list)
            offspring_after_mutation_by_line_id[idx][line] = cur_busses.copy()
        r = random.random()
        if r < 0.4:
            line_num, new_line = create_random_bus_line(system, lines, sizes)
            if line_num not in offspring_after_crossover_by_line_id[idx]:
                offspring_after_crossover_by_line_id[idx][line_num] = new_line
                # offspring_after_mutation_by_line_id[idx][line] = new_line.copy()
        # copy all of the previous lines to the new one
        for line_id in offspring_after_crossover_by_line_id[idx]:
            offspring_after_mutation_by_line_id[idx][line_id] = offspring_after_crossover_by_line_id[idx][
                line_id].copy()

        offspring_after_mutation_by_minute[idx] = defaultdict(list)
        for line in offspring_after_mutation_by_line_id[idx]:
            for minute, size in offspring_after_mutation_by_line_id[idx][line]:
                offspring_after_mutation_by_minute[idx][minute].append((line, size))

    return offspring_after_mutation_by_minute, offspring_after_mutation_by_line_id


def genetic_algorithm(system,
                      lim_seconds: int = 9223372036854775807,
                      with_size_adviser: bool = True,
                      num_generations: int = 100,
                      show_graph: bool = True) -> Tuple[Tuple[Dict[int, List[Tuple[int, int]]],
                                                              Dict[int, List[Tuple[int, int]]]],
                                                        List[int]]:
    """
    apply Genetic algorithm to optimize
    :param system:the system we work in
    :param lim_seconds: optional limit time for the execution
    :param with_size_adviser: optional to use size adviser
    :param num_generations: number of generations
    :param show_graph: if true plot graph of the best solution for each generation
    :return: best solution founded, best fitness of each generation
    """
    best_outputs = []
    start = time.time()
    pop = create_population(system, 300)
    new_pop_size = 50
    # print("stopped")
    # return
    for generation in range(num_generations):

        print("Generation : ", generation)
        start0 = time.time()
        # Measuring the fitness of each chromosome in the population.

        parents = select_mating_pool(system, pop, new_pop_size, with_size_adviser)
        offspring, offspring_bus_lines = crossover([parent[0] for parent in parents], [parent[1] for parent in parents],
                                                   new_pop_size)

        offspring_mutation = mutation(system, offspring, offspring_bus_lines, with_size_adviser)

        pop = parents + list(zip(offspring_mutation[0], offspring_mutation[1]))
        losses = np.empty([len(pop)], dtype=int)
        for i in range(len(pop)):
            losses[i] = total_loss(system, pop[i][0])  # for p in pop
        best_outputs.append(np.min(losses))
        # Getting the best solution after iterating finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        # Then return the index of that solution corresponding to the best fitness.
        if time.time() - start >= lim_seconds:
            break
        print("finished in ", time.time() - start0)
    best_match_idx = int(np.argmin([total_loss(system, p[0]) for p in pop]))

    # print("Best solution : ", pop[best_match_idx][1])
    # print("Best solution fitness : ", fitness[best_match_idx])
    if show_graph:
        plot.title("Genetic Algorithm")
        # plot.xticks(list(range(len(best_outputs))))
        plot.plot(best_outputs)
        plot.xlabel("Generation")
        plot.ylabel("Fitness")
        # tickpos = [range(len(best_outputs))]
        # plot.xticks(tickpos,tickpos)
        plot.show()
    return pop[best_match_idx], best_outputs


def union_sol_of_lines(sol_map: Dict[int, Tuple[Dict[int, List[Tuple[int, int]]],
                                                Dict[int, List[Tuple[int, int]]]]]) \
        -> Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]:
    """
    take solutions for each line and unite them
    :param sol_map: dict of solution for each line
    :return: united solution
    """
    union_sol_min, union_sol_line = defaultdict(list), defaultdict(list)
    for line_id, sol in sol_map.items():
        sol_min, sol_line = sol
        union_sol_line.update(sol_line)
        for minute in sol_min.keys():
            for _, size in sol_min[minute]:
                union_sol_min[minute].append((line_id, size))
    return union_sol_min, union_sol_line


def genetic_algorithm_optimize_lines_sep(system: System,
                                         lim_seconds: int = 9223372036854775807,
                                         with_size_adviser: bool = True,
                                         num_generations_for_each_line: int = 50,
                                         show_graph: bool = True) \
        -> Tuple[Tuple[Dict[int, List[Tuple[int, int]]],
                       Dict[int, List[Tuple[int, int]]]],
                 List[int]]:
    """
    apply Genetic algorithm to optimize but optimize each line separately
    :param system: the system we work in
    :param lim_seconds: optional limit time for the execution
    :param with_size_adviser: optional to use size adviser
    :param num_generations_for_each_line: number of generations for each line
    :param show_graph: if true show graph of the best fitness throughout the running
    :return: best solution founded, best fitness of each generation
    """
    best_sol: Dict[int, Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]] = {}
    best_outputs: List[int] = []
    num_of_lines = len(system.lines())
    for line_id in system.lines():
        print(f'Start optimize line {line_id}')
        system_for_line = system.get_system_of_line(line_id)
        # print(system_for_line.lines())
        (best_sol[line_id], best_outputs_round) = genetic_algorithm(system_for_line, int(lim_seconds / num_of_lines),
                                                                    with_size_adviser=with_size_adviser,
                                                                    num_generations=num_generations_for_each_line,
                                                                    show_graph=False)
        if not best_outputs:
            best_outputs = best_outputs_round
        else:
            size = min(len(best_outputs), len(best_outputs_round))
            temp = []
            for j in range(size):
                temp.append(best_outputs[j] + best_outputs_round[j])
            best_outputs = temp
    if show_graph:
        plot.title(f"Genetic Algorithm- Lines Heuristic")
        # plot.xticks(list(range(len(best_outputs))))
        plot.plot(best_outputs)
        plot.xlabel("Generation")
        plot.ylabel("Fitness")
        # tickpos = [range(len(best_outputs))]
        # plot.xticks(tickpos,tickpos)
        plot.show()

    # best_sol[line_id] = pop[best_match_idx]#(pop[best_match_idx],best_outputs)
    return union_sol_of_lines(best_sol), best_outputs
