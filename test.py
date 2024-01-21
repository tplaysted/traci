from constants import sumoCmd, t_step, total_steps
from evaluation import Evaluator
import pickle
import neat
import os
import statistics
import random
import csv
from multiprocessing import Process, Array
import multiprocessing


def test_winner(config_file, w_path="neat/grid/winner-genome-1"):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE AND TEST WINNER ==== #
    with open(w_path, 'rb') as f:
        winner = pickle.load(f)

    # Watch the winning genome perform
    ev = Evaluator(sumo_cmd=['sumo-gui'] + sumoCmd[1:] + ["--random"],
                   runtime=total_steps)
    net = neat.ctrnn.CTRNN.create(winner, config, t_step)
    return ev.get_net_fitness(net)


def test_baseline():
    ev = Evaluator(sumo_cmd=['sumo-gui'] + sumoCmd[1:] + ["--random"],
                   runtime=total_steps)
    return ev.run_baseline()


def get_stats_auxiliary(array, seeds, config=None, genome=None):
    """20
    An auxiliary evaluation function needed for running simulations in parallel.
    """
    ev = Evaluator(sumo_cmd=sumoCmd, runtime=total_steps)
    fitnesses = [0 for _ in range(len(seeds))]

    if genome is not None:
        if config is None:
            raise ValueError("You need to provide a config if evaluating genomes")
        net = neat.ctrnn.CTRNN.create(genome, config, t_step)
        for i, s in enumerate(seeds):
            fitnesses[i] = ev.get_net_fitness(net, cmd=sumoCmd + ['--seed', str(s)])

            array[i] = fitnesses[i]  # get score, write to shared array
    else:
        for i, s in enumerate(seeds):
            fitnesses[i] = ev.run_baseline(cmd=sumoCmd + ['--seed', str(s)])

            array[i] = fitnesses[i]  # get score, write to shared array


def get_stats_parallel(config=None, num=None, n=100, genome=None):
    if num is None:
        num = multiprocessing.cpu_count()

    chunk_size = n // num
    splits = [[0 for _ in range(chunk_size + 1)] for _ in range(n % num)] \
             + [[0 for _ in range(chunk_size)] for _ in range(num - n % num)]  # split the jobs between processes
    arrays = [Array('d', len(split)) for split in splits]  # each process shares an array with the parent process

    processes = []

    k = 0
    for split in splits:  # distribute genomes
        for i in range(len(split)):
            split[i] = k
            k += 1

    for i in range(num):  # initiate processes
        new_proc = Process(target=get_stats_auxiliary,
                           kwargs={'genome': genome,
                                   'config': config,
                                   'array': arrays[i],
                                   'seeds': splits[i]})
        processes.append(new_proc)
        new_proc.start()

    for proc in processes:  # stop processes
        proc.join()

    output = [0 for _ in range(n)]
    k = 0
    for array in arrays:  # collect all the fitnesses and modify genomes accordingly
        for i in range(len(array)):
            output[k] = array[i]
            k += 1

    return output


def get_genome_stats(config_file, w_path="neat/grid/winner-genome-1", output_path=None, n=100):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE WINNER ==== #
    with open(w_path, 'rb') as f:
        winner = pickle.load(f)

    scores = get_stats_parallel(config, n=n, genome=winner)

    if output_path is not None:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for val in scores:
                writer.writerow([val])

    print("Average score: ", (sum(scores) / n))
    print("Standard deviation: ", statistics.stdev(scores))


def get_baseline_stats(output_path=None, n=100):
    scores = get_stats_parallel(n=n)

    if output_path is not None:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for val in scores:
                writer.writerow([val])

    print("Average score: ", (sum(scores) / n))
    print("Standard deviation: ", statistics.stdev(scores))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat/config-ctrnn')

    # get_genome_stats(config_path, w_path='neat/grid/winner-genome-2',  output_path=None, n=1000)
    get_baseline_stats(n=1000, output_path=None)

    # print(test_winner(config_file=config_path, w_path='neat/grid/winner-genome-2'))
    # print(test_baseline())
