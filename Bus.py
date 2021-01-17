from BusLine import BusLine


class Bus:
    """
    class for bus
    """

    def __init__(self, line: BusLine, exit_time: int, max_size: int) -> None:
        """
        init bus
        :param line: the line id that this bus related to
        :param exit_time: the time this bus leave the first station
        :param max_size: the max amount of passengers can be on the bus at the same time
        """
        self.passengers = []
        self.station_i = 0
        self.current_station_id = line.get_i_station(self.station_i)
        self.time_till_station = line.system.distance_map()[(line.get_i_station(0), line.get_i_station(1))]
        self.max_size = max_size
        self.line = line
        self.counter = 0
        self.max_use = 0
        self.exit_time = exit_time
        self.size_adviser = None

    def update_time(self) -> bool:
        """
        update the position of the bus with one minute
        :return: True if arrived to station o.w false
        """
        if self.time_till_station <= 0:
            self.station_i += 1
            # if we got to the end, we need to stop and remove the bus from
            if self.station_i == len(self.line.station_ids) - 1:
                self.current_station_id = self.line.get_i_station(self.station_i)
                self.leave_passengers()
                self.line.system.remove_bus(self.line, self)
                if self.size_adviser:
                    self.size_adviser.bus_stopped(self)
                return False
            self.time_till_station = self.line.system.distance_map()[
                (self.line.get_i_station(self.station_i), self.line.get_i_station(self.station_i + 1))]
            self.current_station_id = self.line.get_i_station(self.station_i)
            return True
        self.time_till_station -= 1
        return False

    def take_passengers(self, current_time: int) -> None:
        """
        take passengers that wait to this line in the current station
        :param current_time: number between 0-1439 witch represent the time in the day (in minutes)
        """
        # get who's waiting for this bus
        waiting_ppl = self.line.line_pass[self.current_station_id]
        # if we can take all of them
        taken = min(len(waiting_ppl), self.max_size - len(self.passengers))
        self.counter += taken
        # take whichever you can
        for i in range(taken):
            self.passengers.append(waiting_ppl[i])
            # update the people's timer
            waiting_ppl[i].update_got_on(current_time)
        # remove from the station the people who got on
        self.line.take_passengers(self.current_station_id, taken)
        if len(self.passengers) > self.max_use:
            self.max_use = len(self.passengers)

    def leave_passengers(self) -> None:
        """
        leave passengers that the current station is there destiny
        """
        p_remove = []
        # gather them
        for p in self.passengers:
            if p.dest_id == self.current_station_id:
                p_remove.append(p)
        # remove them
        for p in p_remove:
            self.passengers.remove(p)

    def add_size_adviser(self, size_adviser: 'SizeAdviser') -> None:
        """
        add size adviser as listener
        :param size_adviser: the size adviser
        """
        self.size_adviser = size_adviser
