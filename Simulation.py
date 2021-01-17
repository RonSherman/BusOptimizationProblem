import pickle
from collections import defaultdict
from typing import Tuple, List, Dict, Union

from cachetools import cached
from cachetools import LRUCache
from SizeAdviser import SizeAdviser
from System import System


def hash_dict(dic: Dict) -> frozenset:
    """
    convert dict to frozenset for hash use
    :param dic: dict to convert
    :return: frozenset represent the  dict
    """
    return frozenset((a, tuple(b)) for a, b in dic.items())


random_arrival_time = {}


@cached(cache=LRUCache(maxsize=256),
        key=lambda system, bus_map, get_hint=False, more_info=False: (hash_dict(bus_map), get_hint, more_info))
def calc_wait_time(system: System,
                   bus_map: Dict[int, List[Tuple[int, int]]],
                   get_hint: bool = False, more_info: bool = False) -> List[
                    Union[int, dict, Dict[int, List[Tuple[int, int]]]]]:
    """
    get possible solution for the system and simulate it and find the waited time
    :param system: the system of passengers and bus lines
    :param bus_map: the solution we test (dict from minute to list of pair line and size)
    :param get_hint: if its true the function use SizeAdviser and return better solution with most fitted sizes
    :param more_info: if its true the function return also the punishment for passengers that weren't got on any bus
    :return: wait time, number of passengers not picked, optional better solution (get_hint=True), optional the \
    punishment (more_info = True)
    """
    size_adviser = SizeAdviser(system.sizes()) if get_hint else None
    system = pickle.loads(pickle.dumps(system, -1))
    arrival_times = system.wait_times
    # get all the needed minutes- arrival times of passengers and busses
    # min ->
    for minute in range(system.end_day, system.end_day + 24 * 60):
        minute = minute % (24 * 60)
        # passenger by 7, vs bus by 7
        # list of passengers that are arriving at 'minute'
        for passenger in arrival_times[minute]:
            # add them

            system.bus_lines[passenger.wanted_bus].add_passengers([passenger])
        # if bus exited, create it
        new_buses = bus_map[minute]
        if new_buses:
            for bus_line_ids, size in new_buses:
                # create bus
                bus = system.create_bus(system.bus_lines[bus_line_ids], minute, size, size_adviser)
                # take from initial station
                bus.take_passengers(minute)
                # no need to leave passengers, bus is empty
        # check location of buses
        for bus in system.get_current_buses():
            # if arrived
            if bus.update_time():
                bus.leave_passengers()
                bus.take_passengers(minute)
                # take off passengers
                # take new passengers
                # passengers that leave
            # if not calculated,
            # minus 1 from time
            # check if in station, add and remove passengers from station
    minute = system.end_day
    while system.get_current_buses():
        for bus in system.get_current_buses():
            # if arrived
            if bus.update_time():
                bus.leave_passengers()
                bus.take_passengers(minute)
        minute += 1
    # after whole day, calculate the average
    total_wait = 0
    not_picked = {line_id: 0 for line_id in system.lines()}
    punishment = 0
    for val in arrival_times.values():
        for passenger in val:
            if passenger.total_waited_time == -1:
                not_picked[passenger.wanted_bus] += 1
                punishment += 50 * not_picked[passenger.wanted_bus]
                # punishment += 125
            else:
                total_wait += passenger.total_waited_time
    total_wait += punishment
    return_values = [total_wait, not_picked]

    if get_hint:
        return_values.append(size_adviser.get_bus_map())
    if more_info:
        return_values.append(punishment)
    return return_values


@cached(cache=LRUCache(maxsize=1024), key=lambda system, bus_map, get_hint=False: (hash_dict(bus_map), get_hint))
def total_loss(system: System,
               bus_map: Dict[int, List[Tuple[int, int]]],
               get_hint: bool = False,
               alpha: int = 2,
               beta: int = 1) -> Union[Tuple[int, Dict[int, List[Tuple[int, int]]], int], int]:
    """
    calculate the total loss of a solution
    :param system: the system of passengers and bus lines
    :param bus_map: the solution we test (dict from minute to list of pair line and size)
    :param get_hint: if its true the function use SizeAdviser and return better solution with most fitted sizes
    :param alpha: the weight of the waited time in the loss
    :param beta: the weight of the cost of the busses in the loss
    :return: the total loss of the solution, optional better solution and his loss (if get_hint = True)
    """
    # size -> money for size
    cost_per_size = system.sizes_to_cost
    sizes = 0
    for minute in bus_map.values():
        for line in minute:
            sizes += cost_per_size[line[1]]
    # sizes=sum( for line in [minute for minute in]])
    if get_hint:
        time, _, hint = calc_wait_time(system, bus_map, get_hint)
        hint_sizes = 0
        for minute in hint.values():
            for line in minute:
                hint_sizes += cost_per_size[line[1]]
        return alpha * time + beta * sizes, hint, alpha * time + beta * hint_sizes
    return alpha * calc_wait_time(system, bus_map, get_hint)[0] + beta * sizes
