# BusOptimizationProblem

The Bus Problem is the following problem:
Given a system of buslines, stations, passengers and list of possible  sizes for the buses and their price for each ride.
We want to determine the exit time of the buses and their sizes to achive minimal cost and minimal wating time.

In order to solve this problem we wrote code in python which represents the system and simulation for a day.

We write two optimization algorithems Genentic Algorithem (GA) and Simulated Annealing (SA).

## Optimization

The problem is multi objective and doesnt have clear function representation and therefore we wrote simulation that get system and list of buses (minute, size, busline)
and returns the price, the total waited time.

## Running the program

Requirements: Python 3.X

### generate a new random system

Instructions can be found in *create_system* documentation (in SystemGenerator.py)

### save system to file
* for readable file use *write_system_to_file* (in SystemGenerator.py)
* for pickle file use *save_object_to_file* (in SystemGenerator.py)

### read system from file
* for readable file use *read_system_from_file* (in SystemGenerator.py)
* for pickle file use *load_object_from_file* (in SystemGenerator.py)

### solve an input system

* For GA - read *genetic_algorithm\[_optimize_lines_sep\]* documentation (in GeneticAlgorithm.py)
* For SA - read *simulated_annealing\[_optimize_lines_sep\]* documentation (in SimulatedAnneling.py)


## To Be Added

* Multi-Threaded Usage 
* C++ Implementation (fully/partially)
