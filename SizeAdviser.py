from collections import defaultdict
from typing import List, Tuple, Dict

from Bus import Bus


class SizeAdviser:
    """
    This class help us follow the most fitted size of each bus
    track the max number of people on each bus during the simulation
    """
    def __init__(self, sizes: List[int]):
        """
        :param sizes: list of possible sizes for busses
        """
        self.busmap = defaultdict(list)
        self.sizes = sizes

    def bus_stopped(self, bus: Bus) -> None:
        """
        decide for the most fitted size of the bus that given
        :param bus: the bus that stopped
        """
        if bus.max_use > 0:
            for i in range(len(self.sizes)):
                if bus.max_use <= self.sizes[i]:
                    self.busmap[bus.exit_time].append((bus.line.id, self.sizes[i]))
                    return

    def get_bus_map(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        :return: the bus map with the most appropriate sizes for each bus
        """
        return self.busmap
