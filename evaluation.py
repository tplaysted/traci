from constants import sumoCmd, t_step, use_libsumo, total_steps, lock_time, limit_time

if use_libsumo:
    import libsumo as traci
else:
    import traci
import neat
import xml.etree.ElementTree as ET
from enum import Enum
from numpy import argmax


class Direction(Enum):
    NS = 0
    EW = 1


def get_yellow_time_for_lane(lane_id):  # Not using this, default yellow time is 4s
    vmax = traci.lane.getMaxSpeed(lane_id)
    max_decel = 3.048  # max comfortable deceleration is 10ft/s^2 by ITE standards
    return vmax / (2 * max_decel)  # yellow time formula


def set_tls_NS(tlsID):
    traci.trafficlight.setPhase(tlsID, 0)


def set_tls_EW(tlsID):
    traci.trafficlight.setPhase(tlsID, 2)


def get_stat_filename(xmlfile):
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    return root.find('output').find('tripinfo-output').get('value')


class Evaluator:
    """
    Class for determining the fitness of NEAT genomes for a given traffic scenario.
    """

    def __init__(self, sumo_cmd, tlights=None, loops=None, runtime=total_steps):
        # self.stat_filename = get_stat_filename(sumo_cmd[2])
        self.cmd = sumo_cmd

        # ==== Start sumo server and obtain rest of variables ==== #
        traci.start(sumo_cmd)

        if tlights is not None:
            self.tlight_IDs = tlights
        else:
            self.tlight_IDs = traci.trafficlight.getIDList()

        if loops is not None:
            self.loop_IDs = loops
        else:
            self.loop_IDs = traci.inductionloop.getIDList()

        self.runtime = runtime

        self.time_loss = {}

        self.locks = {}

        for k in self.tlight_IDs:
            self.locks[k] = 0

    def __del__(self):
        traci.close()

    def reset(self, cmd=None):
        if cmd is None:
            cmd = self.cmd

        self.time_loss = {}

        for k in self.tlight_IDs:
            self.locks[k] = 0

        traci.load(cmd[1:])

    def do_timestep(self):
        traci.simulationStep()
        self.update_time_loss()

        for tlsID in self.tlight_IDs:
            if self.locks[tlsID] >= limit_time:
                cur_phase = traci.trafficlight.getPhase(tlsID)

                if cur_phase == 1:  # change phase if over time limit
                    set_tls_EW(tlsID)
                    self.locks[tlsID] = 0
                elif cur_phase == 3:
                    set_tls_NS(tlsID)
                    self.locks[tlsID] = 0
            else:
                self.locks[tlsID] += 1

    def run_baseline(self, cmd=None):  # A good baseline function
        if cmd is None:
            self.reset()
        else:
            self.reset(cmd=cmd)

        step = 0

        while step < self.runtime:
            self.do_timestep()
            step += 1

            if traci.simulation.getMinExpectedNumber() == 0:
                break

        num_remaining = traci.vehicle.getIDCount()  # penalise leaving vehicles stranded
        return -1 * (self.get_average_time_loss_fast() + 50 * num_remaining)

    def execute_net_decision(self, net: neat.nn, inputs):
        if len(inputs) != traci.inductionloop.getIDCount():
            raise ValueError("Number of network inputs must match the number of induction loops.")

        if type(net) == neat.ctrnn.CTRNN:  # Continuous Time Recurrent NN (CTRNN) has slightly different implementation
            outputs = net.advance(inputs, t_step, t_step)
        else:
            outputs = net.activate(inputs)

        if len(outputs) != 2 * len(self.tlight_IDs):  # two output nodes per intersection
            raise ValueError("Number of network outputs must match the number of traffic lights under network control.")

        for i, tlsID in enumerate(self.tlight_IDs):
            cur_phase = traci.trafficlight.getPhase(tlsID)
            choice = Direction(argmax(outputs[i * 2: i * 2 + 2]))

            if self.locks[tlsID] <= int(lock_time / t_step):  # implement time lock on recently changed signals
                continue

            if cur_phase == 0 or cur_phase == 2:  # traffic lights that are mid-change are locked
                continue

            if choice == Direction.NS and cur_phase != 1:  # no need to change if already that state
                set_tls_NS(tlsID)
                self.locks[tlsID] = 0
            elif choice == Direction.EW and cur_phase != 3:
                set_tls_EW(tlsID)
                self.locks[tlsID] = 0

    def get_inputs(self):  # Should change to subscription-based polling
        return [traci.inductionloop.getIntervalOccupancy(loopID) for loopID in self.loop_IDs]

    def update_time_loss(self):
        constant = traci.constants.VAR_TIMELOSS
        for veh_id in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(veh_id, [constant])

        for key, value in traci.vehicle.getAllSubscriptionResults().items():
            self.time_loss[key] = value[constant]

    def get_average_time_loss_fast(self):
        total_time_loss = 0
        num_vehicles = len(self.time_loss)
        for key, value in self.time_loss.items():
            total_time_loss += value

        if num_vehicles == 0:
            return 0
        else:
            return total_time_loss / num_vehicles

    def get_median_time_loss_fast(self):
        num_vehicles = len(self.time_loss)
        time_losses = []
        for key, value in self.time_loss.items():
            time_losses.append(value)

        if num_vehicles == 0:
            return 0
        else:
            return time_losses[num_vehicles // 2]

    def get_max_time_loss(self):
        return max(self.time_loss.values())

    def get_net_fitness(self, net: neat.nn, cmd=None):
        if cmd is None:
            self.reset()
        else:
            self.reset(cmd=cmd)

        step = 0
        net.reset()

        while step < self.runtime:
            self.do_timestep()
            self.execute_net_decision(net, self.get_inputs())
            step += 1

            if traci.simulation.getMinExpectedNumber() == 0:
                break

        num_remaining = traci.vehicle.getIDCount()  # penalise leaving vehicles stranded
        return -1 * (self.get_average_time_loss_fast() + 50 * num_remaining)
