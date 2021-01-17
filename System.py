from collections import defaultdict
from copy import deepcopy
from typing import List, Dict, Tuple

from BusLine import BusLine
from Bus import Bus
from Passenger import Passenger
from SizeAdviser import SizeAdviser


class System:
    """
    class of the system we try to optimize
    """

    def __init__(self) -> None:
        """
        create new instance of system
        """
        self.times_between = {}
        self.bus_lines: Dict[int, BusLine] = {}
        # tuple of station id: travel time
        self.map = {}
        # create Passenger list\
        # id -> bus
        self.cur_buses = defaultdict(list)
        self.wait_times = defaultdict(list)
        self.sizes_to_cost = {15: 200, 30: 350, 50: 550}
        self.changed_cur = True
        # self.wait_by_line
        self.cached_cur = []
        self.start_day = 60 * 5
        self.end_day = 60 * 2

    # create bus instance of line
    def create_bus(self, bus_line: BusLine, minute: int, size: int, size_adviser: SizeAdviser) -> Bus:
        """
        create new bus of specific line
        :param bus_line: the line the new bus belong to
        :param minute: the exit time of the new bus
        :param size: the max amount of passengers can be on the bus at the same time
        :param size_adviser: listener to the bus max use (None if not used)
        :return: new Bus with the parameters above
        """
        b = Bus(self.bus_lines[bus_line.id], minute, size)
        if size_adviser:
            b.add_size_adviser(size_adviser)
        self.cur_buses[bus_line].append(b)
        self.changed_cur = True
        return b

    def get_current_buses(self) -> List[Bus]:
        """
        :return: list of the buses that currently drive
        """
        if not self.changed_cur:
            return self.cached_cur
        lis = []
        for val in self.cur_buses.values():
            for bus in val:
                lis.append(bus)
        self.cached_cur = lis
        self.changed_cur = False
        return lis

    def remove_bus(self, bus_line: BusLine, bus: Bus) -> None:
        """
        remove bus that arrive to least station
        :param bus_line: the bus line of the bus
        :param bus: the bus that arrived to least station
        """
        self.cur_buses[bus_line].remove(bus)
        self.changed_cur = True

    def add_bus_lines(self, bus_lines: List[BusLine]) -> None:
        """
        add list of bus_lines to the system
        :param bus_lines: list of bus lines to add
        """
        for bs in bus_lines:
            self.bus_lines[bs.id] = bs

    def add_passengers(self, passengers: List[Passenger]) -> None:
        """
        add list of passengers to the system
        :param passengers: list of passengers
        """
        for ps in passengers:
            self.wait_times[ps.arrival_time].append(ps)
            self.bus_lines[ps.wanted_bus].wait_times[ps.arrival_time].append(ps)

    def sizes(self) -> List[int]:
        """
        :return: list of possible sizes of busses
        """
        return list(self.sizes_to_cost.keys())

    def lines(self) -> List[int]:
        """
        :return: list of the bus lines ids in the system
        """
        return list(self.bus_lines.keys())

    def deepcopy(self) -> 'System':
        """
        :return: deep copy of the system
        """
        copy_system = deepcopy(self)
        return copy_system

    def passengers(self) -> List[Passenger]:
        """
        :return: list of passengers in the system
        """
        ps = []
        for passengers in self.wait_times.values():
            ps += passengers
        return ps

    def get_system_of_line(self, line_id: int) -> 'System':
        """
        return new system that relevant only to the specific line
        :param line_id: the line id
        :return: the system for this line
        """
        new_sys = System()
        bus_line = self.bus_lines[line_id]
        new_bus_line = BusLine(bus_line.station_ids, new_sys, line_id)
        new_sys.add_bus_lines([new_bus_line])
        new_sys.add_distance_map(self.distance_map())
        passengers = self.passengers()
        new_sys.add_passengers([ps for ps in passengers if ps.wanted_bus == line_id])
        return new_sys

    def add_distance_map(self, times_between: Dict[Tuple[int, int], int]) -> None:
        """
        set the distance between stations
        :param times_between: distance map
        """
        self.times_between = times_between

    def distance_map(self) -> Dict[Tuple[int, int], int]:
        """
        :return: the distance map
        """
        return self.times_between
