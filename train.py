import os
import neat
import visualize
from evaluation import Evaluator
import pickle

sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
sumoCmd = [sumoBinary, "-c", "cross.sumocfg"]


def eval_genomes(genomes, config, runs_per_net=4):
    ev = Evaluator(sumo_cmd=sumoCmd, runtime=200, threads=False)
    fitnesses = [[0 for _ in range(runs_per_net)] for _ in range(len(genomes))]
    for i in range(runs_per_net):
        for j, genome in enumerate(genomes):
            net = neat.ctrnn.CTRNN.create(genome[1], config, 1)
            fitnesses[j][i] = ev.get_net_fitness(net)

    for j, genome in enumerate(genomes):
        genome[1].fitness = min(fitnesses[j])


def eval_genome(genome, config):
    ev = Evaluator(sumo_cmd=sumoCmd, runtime=200, threads=True)
    net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
    genome.fitness = ev.get_net_fitness(net)


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== SIMULATION RUN ==== #
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-79')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100))

    # Run for however many generations.
    # pe = neat.ThreadedEvaluator(4, eval_genome)
    # winner = p.run(pe.evaluate, 5)
    # pe.stop()

    winner = p.run(eval_genomes, 50)

    # Save the winner.
    with open('winner-genome', 'wb') as f:
        pickle.dump(winner, f)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', 0: 'OUTPUT'}
    visualize.draw_net(config, winner, False, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=False)
    visualize.plot_species(stats, view=False)


def test_winner(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE AND TEST WINNER ==== #
    with open('winner-genome', 'rb') as f:
        winner = pickle.load(f)

    # Watch the winning genome perform
    ev = Evaluator(sumo_cmd=["C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe", "-c", "cross.sumocfg"],
                   runtime=200)
    net = neat.ctrnn.CTRNN.create(winner, config, 1)
    print(ev.get_net_fitness(net))


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-ctrnn')

    # run(config_path)
    test_winner(config_path)
