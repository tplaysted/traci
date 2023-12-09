import traci
import neat
import xml.etree.ElementTree as ET
import threading


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


def update_config_for_threading(xmlfile, thread):  # Will probably delete this
    tree = ET.parse(xmlfile)
    root = tree.getroot()
    tripinfo = root.find('output').find('tripinfo-output')
    new_value = tripinfo.get('value')[:-4] + '_' + str(thread) + '.xml'
    new_filename = xmlfile[:-8] + '_' + str(thread) + '.sumocfg'

    tripinfo.set('value', new_value)
    tree.write(new_filename)

    return new_filename


class Evaluator:
    """
    Class for determining the fitness of NEAT genomes for a given traffic scenario.
    """

    def __init__(self, sumo_cmd, tlights=None, loops=None, runtime=1000, threads=False):
        # ==== Do prep for multithreading ==== #
        thread = threading.get_ident()

        if threads:
            new_cfg = update_config_for_threading(sumo_cmd[2], thread)
            self.stat_filename = get_stat_filename(new_cfg)
            self.cmd = [sumo_cmd[0], sumo_cmd[1], new_cfg]
        else:
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

    def __del__(self):
        traci.close()

    def reset(self):
        self.time_loss = {}
        traci.load(['-c', self.cmd[2]])

    def run_baseline(self):  # A good baseline function
        step = 0
        while step < self.runtime:
            if step % self.runtime == self.runtime // 2:
                set_tls_NS('0')
            elif step % self.runtime == 0:
                set_tls_EW('0')

            traci.simulationStep()
            self.update_time_loss()

            step += 1

        return -1 * self.get_max_time_loss()

    def execute_net_decision(self, net: neat.nn, inputs):
        if type(net) == neat.ctrnn.CTRNN:  # Continuous Time Recurrent NN (CTRNN) has slightly different implementation
            outputs = net.advance(inputs, 1, 1)
        else:
            outputs = net.activate(inputs)

        if len(outputs) != len(self.tlight_IDs):
            raise ValueError("Number of network outputs must match the number of traffic lights under network control.")

        for i, tlsID in enumerate(self.tlight_IDs):
            cur_phase = traci.trafficlight.getPhase(tlsID)

            if cur_phase == 0 or cur_phase == 2:  # traffic lights that are mid-change are locked
                continue

            if outputs[i] <= 0.5 and cur_phase != 1:  # no need to change if already that state
                set_tls_NS(tlsID)
            elif outputs[i] > 0.5 and cur_phase != 3:
                set_tls_EW(tlsID)

    def get_inputs(self):  # Should change to subscription-based polling
        return [traci.inductionloop.getIntervalOccupancy(loopID) for loopID in self.loop_IDs]

    def update_time_loss(self):
        constant = traci.constants.VAR_TIMELOSS
        for veh_id in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(veh_id, [constant])

        for key, value in traci.vehicle.getAllSubscriptionResults().items():
            self.time_loss[key] = value[constant]

    def get_average_time_loss(self):
        # We get the timeLoss of arrived vehicles by parsing the live_stats output file, since the vehicle objects
        # no longer exist
        with open('sumo/' + self.stat_filename, 'a') as fp:
            fp.write('</tripinfos>\n')

        tree = ET.parse('sumo/' + self.stat_filename)
        root = tree.getroot()
        total_time_loss = 0
        num_vehicles = 0

        for veh in root.findall('tripinfo'):
            total_time_loss += float(veh.get('timeLoss'))
            num_vehicles += 1

        with open('sumo/' + self.stat_filename, 'r+') as fp:
            lines = fp.readlines()
            fp.seek(0)
            fp.truncate()
            fp.writelines(lines[:-1 * (num_vehicles + 1)])  # completely reset the file -1 * (num_vehicles + 1)]

        # We need to also count vehicles still in the simulation

        vehicles = traci.vehicle.getIDList()

        for veh in vehicles:
            total_time_loss += traci.vehicle.getTimeLoss(veh)
            num_vehicles += 1

        if num_vehicles == 0:
            return 0

        return total_time_loss / num_vehicles

    def get_average_time_loss_fast(self):
        total_time_loss = 0
        num_vehicles = len(self.time_loss)
        for key, value in self.time_loss.items():
            total_time_loss += value

        if num_vehicles == 0:
            return 0
        else:
            return total_time_loss / num_vehicles

    def get_max_time_loss(self):
        return max(self.time_loss.values())

    def get_net_fitness(self, net: neat.nn):
        step = 0
        net.reset()
        self.reset()

        while step < self.runtime:
            traci.simulationStep()
            self.update_time_loss()
            self.execute_net_decision(net, self.get_inputs())
            step += 1

            if traci.simulation.getMinExpectedNumber() == 0:
                break

        return -1 * (self.get_max_time_loss() + self.get_average_time_loss_fast())
