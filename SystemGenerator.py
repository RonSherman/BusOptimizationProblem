# creates solution
import math
import pickle
import random
from collections import defaultdict

from typing import Tuple, List, Dict

from BusLine import BusLine
from Passenger import Passenger
from System import System

random_arrival_time = {}


def get_random_arrival_time(system: System, bl_id: int) -> int:
    """
    this function is heuristic for when the busses should exit
    :param system: the system we work in
    :param bl_id: the id of the line we want get new exit time
    :return: random exit time from good options
    """
    minutes = []
    if bl_id in random_arrival_time:
        return random.choice(random_arrival_time[bl_id])
    # for each bus line,
    bl = system.bus_lines[bl_id]
    for minute, passengers in bl.wait_times.items():
        for ps in passengers:
            minutes.append((minute - bl.time_till_ps_station(ps.start_id) + 24 * 60) % (24 * 60))
    random_arrival_time[bl_id] = minutes
    return random.choice(minutes)


def create_random_bus(system: System, bl_id: int, sizes: List[int]) -> Tuple[int, int]:
    """
    create random bus for specific line
    :param system: the system we work in
    :param bl_id: the line id
    :param sizes: the possible sizes
    :return: the bus exit time and size
    """
    minute = get_random_arrival_time(system, bl_id)  # random.randint(0,24*60-1)
    # minute = int(min(64*24-1,max(0,random.normalvariate(60*24/2,math.sqrt(50*60*24/2)))))
    size = random.choice(sizes)
    return minute, size


def create_random_bus_line(system: System, lines: List[int], sizes: List[int], num_of_busses: int = 1) \
        -> Tuple[int, List[Tuple[int, int]]]:
    """
    create random bus line with some buses
    :param system: the system we work in
    :param lines: the possible lines
    :param sizes: the possible sizes
    :param num_of_busses: the number of buses to create for this bus line
    :return: the line id and the new buses (minutes, size)
    """
    line_map = []
    line = random.choice(lines)
    for i in range(num_of_busses):
        minute, size = create_random_bus(system, line, sizes)
        line_map.append((minute, size))
    return line, line_map


def create_random_bas_map(system: System) -> Dict[int, List[Tuple[int, int]]]:
    """
    create random solution
    :param system: the system we work in
    :return: bus map
    """
    bus_map_by_line = defaultdict(list)
    lines = system.lines()
    sizes = system.sizes()
    for i in range(int(2 * len(lines) * max(1, int(0.5 + math.log(len(lines)))))):
        line_num, new_line = create_random_bus_line(system, lines, sizes, random.randint(1, 5))
        if line_num not in bus_map_by_line:
            bus_map_by_line[line_num] = new_line
    return bus_map_by_line


def line_bus_map_to_min(bus_map_by_line: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    convert dict of line to list of min,size to dict of min to list of line, size
    :param bus_map_by_line: the dict to convert
    :return: the new dict
    """
    bus_map_by_min = defaultdict(list)
    for line in bus_map_by_line:
        for minute, size in bus_map_by_line[line]:
            bus_map_by_min[minute].append((line, size))
    return bus_map_by_min


def min_bus_map_to_line(bus_map_by_min: Dict[int, List[Tuple[int, int]]]) -> Dict[int, List[Tuple[int, int]]]:
    """
    convert dict of min to list of line, size to dict of line to list of min,size
    :param bus_map_by_min: the dict to convert
    :return: the new dict
    """
    bus_map_by_line = defaultdict(list)
    for minute in bus_map_by_min:
        for line, size in bus_map_by_min[minute]:
            bus_map_by_line[line].append((minute, size))
    return bus_map_by_line


def create_population(system: System, start_population: int = 50) -> \
        List[Tuple[Dict[int, List[Tuple[int, int]]], Dict[int, List[Tuple[int, int]]]]]:
    """
    create n random solutions
    :param system: the system we work in
    :param start_population: the number of random solutions to generate
    :return: list of the solutions created
    """
    population = []
    for i in range(start_population):
        bus_map_by_line = create_random_bas_map(system)
        bus_map_by_min = line_bus_map_to_min(bus_map_by_line)
        population.append((bus_map_by_min, bus_map_by_line))
    return population


def generate_lines(num_lines: int, num_stations: int, system: System) -> Tuple[List[BusLine], List[int]]:
    """
    create n bus lines
    :param num_lines: the number of random bus lines to generate
    :param num_stations: the number of random stations to generate
    :param system: the system we work in
    :return: the new bus lines, stations
    """
    stations_id = list(range(1, num_stations + 1))
    # decide on travel distance
    distances = {}
    lines = []
    for i in range(num_lines):
        line_stations = random.sample(stations_id, random.randint(min(5, max(num_stations // 2, 2)), num_stations))
        for j in range(len(line_stations) - 1):
            if (line_stations[j], line_stations[j + 1]) not in distances:
                distances[(line_stations[j], line_stations[j + 1])] = random.randint(5, 25)
                # distances[(line_stations[j+1], line_stations[j])] = \
                # distances[(line_stations[j], line_stations[j+1])] for reverse lines
        line_id = i * 10 + random.randint(0, 9)
        lines.append(BusLine(line_stations, system, line_id))
        # lines.append(BusLine(list(reversed(line_stations)),distances,system,str(line_id)+'R')) for reverse lines
    system.add_distance_map(distances)
    return lines, stations_id


# generate a good passengers
def generate_passengers(system: System, number_of_peoples_per_line: int = 750) -> List[Passenger]:
    """
    create n passengers for each bus line
    :param system: the system we work in
    :param number_of_peoples_per_line: the number of passengers to generate for each line
    :return: the new passengers
    """
    passengers = []
    # for each bus line, around 750 ppl a day ride it
    ppl_per_bus_line = number_of_peoples_per_line

    for bl in system.bus_lines.values():
        for i in range(ppl_per_bus_line):
            # random.normalvariate(cur_minute, 60)
            r = random.random()
            # morning time- around 8:00 AM
            if r < 0.48:
                rand_min = max(60 * 5, int(random.normalvariate(60 * 8, 69)))
                if rand_min == 60 * 5:
                    rand_min = random.randint(60 * 5, 60 * 5 + 30)
                # random.choice(range(0,1440))
            # evening time- around 18:00
            elif r < 0.96:
                rand_min = int(random.normalvariate(1080, 120))
            else:
                # between 2AM-5AM theres no busses
                rand_min = random.choice([random.randint(0, 60 * 2 - 1), random.randint(60 * 5, 60 * 24 - 1)])
            start_index_station = random.choice(range(len(bl.station_ids) - 1))
            start_station_id = bl.station_ids[start_index_station]
            end_station_id = random.choice(bl.station_ids[start_index_station + 1:])
            passengers.append(Passenger(start_station_id, end_station_id, bl.id, rand_min))
    return passengers


def read_system_from_file(file_name: str) -> System:
    """
    create System from file
    :param file_name: the name of the file with the data on the system
    :return: the system
    """
    system = System()
    bus_lines = []
    ps = []
    with open(file_name, 'r') as file_system:
        stations_ids = file_system.readline().strip().split(',')
        dist_map = {}
        line = file_system.readline().strip()
        while line:
            xy, dist = line.split(':')
            x, y = xy.split(',')
            dist_map[(int(x), int(y))] = int(dist)
            line = file_system.readline().strip()
        line = file_system.readline().strip()
        while line:
            line_id, stations = line.split(':')
            line_stations = stations.split(',')
            bus_lines.append(BusLine(list(map(int,line_stations)), system, int(line_id)))
            line = file_system.readline().strip()
        line = file_system.readline().strip()
        while line:
            minute, wanted_line, start, dest = line.split(',')
            ps.append(Passenger(int(start), int(dest), int(wanted_line), int(minute)))
            line = file_system.readline().strip()
    system.add_bus_lines(bus_lines)
    system.add_passengers(ps)
    system.add_distance_map(dist_map)
    return system


def write_system_to_file(system: System, file_name: str) -> None:
    """
    create System from file
    :param system: the system to save
    :param file_name: the name of the file with the data on the system
    """
    stations_ids = set()
    dists = []
    for k, v in system.distance_map().items():
        stations_ids.update(k)
        dists.append(f'{k[0]},{k[1]}:{v}\n')
    stations_ids = [f"{','.join(list(map(str, stations_ids)))}\n"]

    bus_lines = [f"{line}:{','.join(list(map(str, system.bus_lines[line].station_ids)))}\n" for line in system.lines()]
    passengers = [f'{ps.arrival_time}, {ps.wanted_bus}, {ps.start_id}, {ps.dest_id}\n' for ps in system.passengers()]
    with open(file_name, 'w') as file_system:
        file_system.writelines(stations_ids)
        file_system.writelines(dists)
        file_system.write('\n')
        file_system.writelines(bus_lines)
        file_system.write('\n')
        file_system.writelines(passengers)


def create_system(number_of_lines: int, number_of_stations: int, number_of_peoples_per_line: int) -> System:
    """
    create new System
    :param number_of_lines: number of lines in the system
    :param number_of_stations: number of stations in the system
    :param number_of_peoples_per_line: number of passengers for each line in the system
    :return: the new system
    """
    system = System()
    lines, stations_id = generate_lines(number_of_lines, number_of_stations, system)
    system.add_bus_lines(lines)
    ps = generate_passengers(system, number_of_peoples_per_line)
    system.add_passengers(ps)
    return system


def save_object_to_file(object_to_save: object, file_name: str) -> None:
    """
    save object to file
    :param object_to_save: object to save
    :param file_name: the name of the file
    """
    with open(file_name, 'wb') as output:
        pickle.dump(object_to_save, output)


def load_object_from_file(file_name: str) -> object:
    """
        read object from file
        :param file_name: the name of the file
        :return: the object
        """
    with open(file_name, 'rb') as input_file:
        obj = pickle.load(input_file)
    return obj
