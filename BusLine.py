from collections import defaultdict
from typing import List, Tuple, Dict


class BusLine:
    """
    class for bus line
    """

    # number of stations ids, time between each
    def __init__(self, station_ids: List[int], times_between: Dict[Tuple[int, int], int], system: 'System',
                 line_id: int) -> None:
        """
        :param station_ids: the stations that this line go throw
        :param times_between: the distance between each station in the route
        :param system: the system this bus line belong to
        :param line_id: id for this line
        """
        self.system = system
        self.station_ids = station_ids
        self.times_between = times_between
        self.id = line_id
        # CURRENT id station, list of people waiting for this line

        # initializing deque
        # de = collections.deque
        self.line_pass = defaultdict(list)  # collections.deque)
        # a map of min->[ps]
        self.wait_times = defaultdict(list)

    def get_i_station(self, i: int) -> int:
        """
        :param i: relative index of the wanted station
        :return: the id of the i-th station
        """
        return self.station_ids[i]

    # new passengers arrived
    def add_passengers(self, passengers: List['Passenger']) -> None:
        """
        add passengers that wait for this line
        :param passengers: list of passengers to add
        """
        # add each ps by its station
        for ps in passengers:
            self.line_pass[ps.start_id].append(ps)

    # passengers aboard a bus,
    def take_passengers(self, station: int, num_taken: int) -> None:
        """
        remove passengers that got on a bus
        :param station: the station where the passengers got on
        :param num_taken: the number of passengers got on
        """
        for i in range(num_taken):
            self.line_pass[station].pop(0)  # popleft()

    def time_till_ps_station(self, stat_id: int) -> int:
        """
        calc time that take to bus of this line to arrive to specific station from the first station
        :param stat_id: the id of the wanted station
        :return: the time that take to arrive from first station to the wanted station (in minutes)
        """
        time = 0
        # cur_stat=stat_id
        flag = False
        for i, sa in enumerate(self.station_ids):
            if sa == stat_id:
                flag = True
                break
            time += self.times_between[(self.station_ids[i], self.station_ids[i + 1])]
        if not flag:
            raise Exception("A case that will never happen- Yehonatan harmatz")
        return time
