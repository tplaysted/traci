from evaluation import Evaluator
import pickle
import neat
import os
import statistics
import random

sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
sumoCmd = [sumoBinary, "-c", "sumo/cross/cross.sumocfg"]


def test_winner(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE AND TEST WINNER ==== #
    with open('neat/winner-genome-1', 'rb') as f:
        winner = pickle.load(f)

    # Watch the winning genome perform
    ev = Evaluator(sumo_cmd=['sumo-gui', "-c", "sumo/cross/cross.sumocfg", "--random"],
                   runtime=200)
    net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)
    return ev.get_net_fitness(net)


def test_baseline():
    ev = Evaluator(sumo_cmd=sumoCmd,
                   runtime=200)
    return ev.run_baseline()


def get_genome_stats(config_file, w_path="neat/winner-genome-1", output_path="data/winner_stats.csv", n=100):
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
    net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)

    scores = [0 for _ in range(n)]

    for i in range(n):
        scores[i] = ev.get_net_fitness(net, cmd=sumoCmd + ['--seed', str(random.randrange(1000))])

    print("Average score: ", (sum(scores) / n))
    print("Standard deviation: ", statistics.stdev(scores))


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat/config-ctrnn')

    # get_genome_stats(config_path, n=1000)

    test_winner(config_file=config_path)
