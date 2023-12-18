from evaluation import Evaluator
import pickle
import neat
import os
import statistics
import random
import csv

sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
sumoCmd = [sumoBinary, "-c", "sumo/grid/grid.sumocfg"]


def test_winner(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE AND TEST WINNER ==== #
    with open('neat/grid/winner-genome', 'rb') as f:
        winner = pickle.load(f)

    # Watch the winning genome perform
    ev = Evaluator(sumo_cmd=['sumo-gui', "-c", "sumo/grid/grid.sumocfg", "--random"],
                   runtime=200)
    net = neat.ctrnn.CTRNN.create(winner, config, 1)
    return ev.get_net_fitness(net)


def test_baseline(cmd=None):
    ev = Evaluator(sumo_cmd=cmd,
                   runtime=200)
    return ev.run_baseline()


def get_genome_stats(config_file, w_path="neat/grid/winner-genome", output_path="data/winner_stats.csv", n=100):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE WINNER ==== #
    with open(w_path, 'rb') as f:
        winner = pickle.load(f)

    # Set up network and evaluator object
    ev = Evaluator(sumo_cmd=sumoCmd,
                   runtime=200)
    net = neat.ctrnn.CTRNN.create(winner, config, 1)

    scores = [0 for _ in range(n)]

    for i in range(n):
        scores[i] = -1 * ev.get_net_fitness(net, cmd=sumoCmd + ['--seed', str(random.randrange(1000))])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for val in scores:
            writer.writerow([val])

    print("Average score: ", (sum(scores) / n))
    print("Standard deviation: ", statistics.stdev(scores))


def get_baseline_stats(output_path="data/baseline_stats.csv", n=100):
    # Set up network and evaluator object
    ev = Evaluator(sumo_cmd=sumoCmd,
                   runtime=200)

    scores = [0 for _ in range(n)]

    for i in range(n):
        scores[i] = -1 * ev.run_baseline(cmd=sumoCmd + ['--seed', str(random.randrange(1000))])

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for val in scores:
            writer.writerow([val])

    print("Average score: ", (sum(scores) / n))
    print("Standard deviation: ", statistics.stdev(scores))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat/config-ctrnn')

    get_genome_stats(config_path, n=1000)
    # get_baseline_stats(n=1000)

    # print(test_winner(config_file=config_path))
    # print(test_baseline(cmd=['sumo-gui', "-c", "sumo/grid/grid.sumocfg", "--random"]))
