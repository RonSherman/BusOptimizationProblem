from EvaluatingAlgorithms import compare_algorithms, more_information
from GeneticAlgorithm import genetic_algorithm, genetic_algorithm_optimize_lines_sep
from SimulatedAnneling import simulated_annealing, simulated_annealing_optimize_lines_sep
from SystemGenerator import create_system, save_object_to_file, load_object_from_file

system = load_object_from_file('System (15,30,750)')
lim_time = 60*10
sols, best_for_each_iter = compare_algorithms(system, [
                                                       simulated_annealing_optimize_lines_sep,
                                                       genetic_algorithm_optimize_lines_sep], lim_seconds=lim_time)


save_object_to_file(sols,f'sols - System (15,30,750) - {lim_time} sec without size advisor')
save_object_to_file(best_for_each_iter,f'best_for_each_iter - System (15,30,750) - {lim_time} sec without size advisor')
print("Total Time Waited,number of people that didn't get on a bus,for each line,punishment for no arriving, Avg Time Waited")
for sol in sols:
    print(more_information(system,sol))

