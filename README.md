# BusOptimizationProblem

The Bus Problem is the following problem:
Given a system of buslines, stations, passengers and list of possible  sizes for the buses and thier price for each ride.
We want to determine the exit time of the buses and their sizes to achive minimal cost and minimal wating time.

In order to solve this problem we wrote code in python which represents the system and simulation for a day.

We write two optimization algorithems Genentic Algorithem (GA) and Simulated Annealing (SA).

## Optimization

The problem is multi objective and doesnt have close function and therfore we wrote simulation that get system and list of buses (minute, size, busline)
and return the price, the total waited time.

## Run the program: options to run

requirements: python version 3

### generate a new random system

read create_system documentation (in SystemGenerator.py)

### save system to file
* for readable file use write_system_to_file
* for pickle file use save_object_to_file

### read system from file
* for readable file use read_system_from_file
* for pickle file use load_object_from_file

### solve an input system

* for GA - read genetic_algorithm\[_optimize_lines_sep\] documentation
* for SA - read simulated_annealing\[_optimize_lines_sep\] documentation
