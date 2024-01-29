t_step = 0.2  # simulation time_step for sumo and ctrnn
runtime = 400
sumoCmd = ['sumo', '-c', 'sumo/cbd/tinycbd.sumocfg', '--step-length', str(t_step)]
use_libsumo = False  # simulations run faster if true, but gui only works if false
total_steps = int(runtime / t_step)
lock_time = 8  # lock time in seconds
limit_time = 50  # maximum time before lights change automatically
