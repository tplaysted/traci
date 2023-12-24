from constants import sumoCmd, t_step, total_steps
from evaluation import Evaluator
import pickle
import neat
import os
import statistics
import random
import csv


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


def get_genome_stats(config_file, w_path="neat/grid/winner-genome-1", output_path=None, n=100):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE WINNER ==== #
    with open(w_path, 'rb') as f:
        winner = pickle.load(f)

    # Set up network and evaluator object
    ev = Evaluator(sumo_cmd=sumoCmd,
                   runtime=total_steps)
    net = neat.ctrnn.CTRNN.create(winner, config, t_step)

    scores = [0 for _ in range(n)]

    for i in range(n):
        scores[i] = -1 * ev.get_net_fitness(net, cmd=sumoCmd + ['--seed', str(i)])

    if output_path is not None:
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for val in scores:
                writer.writerow([val])

    print("Average score: ", (sum(scores) / n))
    print("Standard deviation: ", statistics.stdev(scores))


def get_baseline_stats(output_path="data/baseline_stats.csv", n=100):
    # Set up network and evaluator object
    ev = Evaluator(sumo_cmd=sumoCmd,
                   runtime=total_steps)

    scores = [0 for _ in range(n)]

    for i in range(n):
        scores[i] = -1 * ev.run_baseline(cmd=sumoCmd + ['--seed', str(i)])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for val in scores:
            writer.writerow([val])

    print("Average score: ", (sum(scores) / n))
    print("Standard deviation: ", statistics.stdev(scores))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat/config-ctrnn')

    get_genome_stats(config_path, w_path='neat/grid/winner-genome-2',  output_path="winner_stats-2", n=1000)
    # get_baseline_stats(n=1000, output_path="data/baseline_stats.csv")

    # print(test_winner(config_file=config_path, w_path='neat/grid/winner-genome-2'))
    # print(test_baseline())
