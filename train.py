import multiprocessing
import os
import neat
import visualize
from evaluation import Evaluator
import pickle
from multiprocessing import Process, Array
import statistics

sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"  # <== Must point to your sumo binary
sumoCmd = [sumoBinary, "-c", "sumo/cross/cross.sumocfg", "-b", "0"]


def eval_genomes(genomes, config, runs_per_net=20):
    ev = Evaluator(sumo_cmd=sumoCmd, runtime=200, threads=False)
    fitnesses = [[0 for _ in range(runs_per_net)] for _ in range(len(genomes))]
    for i in range(runs_per_net):  # Testing all the genomes
        for j, genome in enumerate(genomes):
            net = neat.ctrnn.CTRNN.create(genome[1], config, 1)
            fitnesses[j][i] = ev.get_net_fitness(net)

    for j, genome in enumerate(genomes):
        genome[1].fitness = min(fitnesses[j]) + max(fitnesses[j])  # get worst score


def eval_genome(genome, config):  # potentially needed for multithreading
    ev = Evaluator(sumo_cmd=sumoCmd, runtime=200, threads=False)
    net = neat.ctrnn.CTRNN.create(genome, config, 1)
    genome.fitness = ev.get_net_fitness(net)


def eval_genomes_auxiliary(genomes, config, array, runs_per_net=1):
    """
    An auxiliary evaluation function needed for running simulations in parallel.
    """
    ev = Evaluator(sumo_cmd=sumoCmd, runtime=200, threads=False)
    fitnesses = [[0 for _ in range(runs_per_net)] for _ in range(len(genomes))]
    for i in range(runs_per_net):  # Testing all the genomes
        for j, genome in enumerate(genomes):
            net = neat.nn.recurrent.RecurrentNetwork.create(genome[1], config)
            fitnesses[j][i] = ev.get_net_fitness(net)

    for j in range(len(array)):
        array[j] = sum(fitnesses[j]) / runs_per_net  # get score, write to shared array


def eval_genomes_parallel(genomes, config, num=None):
    if num is None:
        num = multiprocessing.cpu_count()

    n = len(genomes)
    chunk_size = n // num
    splits = [[None for _ in range(chunk_size + 1)] for _ in range(n % num)] \
             + [[None for _ in range(chunk_size)] for _ in range(num - n % num)]  # split the jobs between processes
    arrays = [Array('d', len(split)) for split in splits]  # each process shares an array with the parent process

    processes = []

    k = 0
    for split in splits:  # distribute genomes
        for i in range(len(split)):
            split[i] = genomes[k]
            k += 1

    for i in range(num):  # initiate processes
        new_proc = Process(target=eval_genomes_auxiliary,
                           kwargs={'genomes': splits[i],
                                   'config': config,
                                   'array': arrays[i]})
        processes.append(new_proc)
        new_proc.start()

    for proc in processes:  # stop processes
        proc.join()

    k = 0
    for array in arrays:  # collect all the fitnesses and modify genomes accordingly
        for i in range(len(array)):
            genomes[k][1].fitness = array[i]
            k += 1


def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== SIMULATION RUN ==== #
    # Create the population, which is the top-level object for a NEAT run.
    # p = neat.Population(config)
    p = neat.Checkpointer.restore_checkpoint('neat/neat-checkpoint-99')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(100, filename_prefix='neat/neat-checkpoint-'))

    # Run for however many generations.
    winner = p.run(eval_genomes_parallel, 100)

    # Save the winner.
    with open('neat/winner-genome', 'wb') as f:
        pickle.dump(winner, f)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', 0: 'OUTPUT'}
    visualize.draw_net(config, winner, False, node_names=node_names, filename='neat/Digraph')
    visualize.plot_stats(stats, ylog=False, view=False, filename='neat/avg_fitness.svg')
    visualize.plot_species(stats, view=False, filename='neat/speciation.svg')


def test_winner(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # ==== RESTORE AND TEST WINNER ==== #
    with open('neat/winner-genome', 'rb') as f:
        winner = pickle.load(f)

    # Watch the winning genome perform
    ev = Evaluator(sumo_cmd=["C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe", "-c", "sumo/cross/cross.sumocfg"],
                   runtime=200)
    net = neat.nn.recurrent.RecurrentNetwork.create(winner, config)
    print(ev.get_net_fitness(net))


def test_baseline(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    ev = Evaluator(sumo_cmd=["C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe", "-c", "sumo/cross/cross.sumocfg"],
                   runtime=200)

    print(ev.run_baseline())


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat/config-ctrnn')

    # run(config_path)
    test_winner(config_path)
    # test_baseline(config_path)
