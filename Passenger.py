class Passenger:
    """
    class of passenger
    """

    def __init__(self, start_id: int, dest_id: int, wanted_bus: int, arrival_time: int) -> None:
        """
        :param start_id: id of the first station where the passenger wait
        :param dest_id: the destination station
        :param wanted_bus: the line the passenger wait for
        :param arrival_time: the time the passenger arrive to the station
        """
        self.start_id = start_id
        self.dest_id = dest_id
        self.wanted_bus = wanted_bus
        self.total_waited_time = -1
        self.arrival_time = arrival_time

    def update_got_on(self, current_time) -> None:
        """
        update the waited time when got on the bus
        :param current_time: current time when the passenger got on the bus
        """
        self.total_waited_time = (60 * 24 + current_time - self.arrival_time) % (60 * 24)

    def update_got_off(self, current_time) -> None:
        """
        update the time the passenger got to the next station
        :param current_time: current time when the passenger got off the bus
        """
        # waiting again
        self.arrival_time = current_time
