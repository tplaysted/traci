from constants import sumoCmd, t_step, total_steps
import multiprocessing
import os
import neat
import visualize
import evaluation
import pickle
from multiprocessing import Process, Array


def eval_genomes(genomes, config, runs_per_net=1):
    ev = evaluation.Evaluator(sumo_cmd=sumoCmd, runtime=total_steps)
    fitnesses = [[0 for _ in range(runs_per_net)] for _ in range(len(genomes))]
    for i, genome in enumerate(genomes):
        for j in range(runs_per_net):
            net = neat.ctrnn.CTRNN.create(genome[1], config, t_step)
            fitnesses[i][j] = ev.get_net_fitness(net, cmd=sumoCmd + ['--seed', str(j)])

        genome[1].fitness = sum(fitnesses[i]) / runs_per_net  # get score


def eval_genomes_auxiliary(genomes, config, array, runs_per_net=20):
    """20
    An auxiliary evaluation function needed for running simulations in parallel.
    """
    ev = evaluation.Evaluator(sumo_cmd=sumoCmd, runtime=total_steps)
    fitnesses = [[0 for _ in range(runs_per_net)] for _ in range(len(genomes))]
    for i, genome in enumerate(genomes):
        net = neat.ctrnn.CTRNN.create(genome[1], config, t_step)
        for j in range(runs_per_net):
            fitnesses[i][j] = ev.get_net_fitness(net, cmd=sumoCmd + ['--seed', str(j)])

        array[i] = sum(fitnesses[i]) / runs_per_net  # get score, write to shared array


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
    p = neat.Population(config)
    # p = neat.Checkpointer.restore_checkpoint('neat/grid/checkpoints/neat-checkpoint-398')

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(100, filename_prefix='neat/grid/checkpoints/neat-checkpoint-'))

    # Run for however many generations.
    winner = p.run(eval_genomes_parallel, 100)

    # Save the winner.
    with open('neat/cbd/winner-genome-6', 'wb') as f:
        pickle.dump(winner, f)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    visualize.draw_net(config, winner, False, filename='neat/cbd/Digraph-6')
    visualize.plot_stats(stats, ylog=False, view=False, filename='neat/cbd/avg_fitness-6.svg')
    visualize.plot_species(stats, view=False, filename='neat/cbd/speciation-6.svg')


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'neat/config-ctrnn-cbd')

    run(config_path)
