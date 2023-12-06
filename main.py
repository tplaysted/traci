import traci
import random
import evaluation as ev

sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"
sumoCmd = [sumoBinary, "-c", "cross.sumocfg"]


if __name__ == "__main__":
    evl = ev.Evaluator(sumo_cmd=sumoCmd)
    ev.run()
    print(evl.get_average_time_loss())

    pass

