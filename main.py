import argparse
import os
import random

import threading
import time
from datetime import datetime, timezone
from sys import stderr

import numpy as np
import yaml
from elasticsearch import Elasticsearch

import battery_controller
import model_utils
import modbus_simul_utils

TIME_QUANT = 5
TIME_SCALE = 1

POWER_REFS_LOG = 'POWER_REFS_LOG',
STATES_LOG = 'STATES_LOG',
COMMON_LOG = 'COMMON_LOG',
BATTERY_LOG = 'BATTERY_LOG',
OTHERS_LOG = 'OTHERS_LOG'


class Logger:
    def __init__(self, appenders):
        self._appenders = appenders

    def log(self, dest, *args, **kwargs):
        np_log(self._appenders[dest], *args, **kwargs)


def np_log(filename, array):
    with open(filename, 'ab') as fout:
        X = np.hstack((time.time(), array))
        np.savetxt(fout, X, newline=',')
        fout.write(b'\n')


def send_to_elastic(elastic, model, wind_power, battery_power):
    doc = dict()

    doc['timestamp'] = datetime.now(timezone.utc).isoformat()
    doc['power_battery'] = battery_power
    doc['power_wind'] = wind_power
    doc['battery_energy'] = model.battery.get_energy()
    doc['wind_energy'] = model.wind_gen.get_energy()
    doc['wind_speed'], doc['temperature'] = model.get_weather_at_time()

    for i, b in enumerate(model.buildings):
        doc[f'heater_{1 + i}_energy'] = b.get_energy()
        doc[f'heater_{1 + i}_power'] = b.current_power
        doc[f'heater_{1 + i}_temperature'] = b.get_temp()

    doc['battery_soc'] = model.battery.get_soc()
    doc['wind_unused_energy'] = model.wind_burner.get_energy()
    doc['diesel_energy'] = model.gen_set.get_energy()
    doc['wind_unused_power'] = model.wind_burner.current_power
    doc['diesel_power'] = model.gen_set.current_power

    elastic.index(index='model', body=doc)


class ThreadSend(threading.Thread):

    def __init__(self, run_event, elastic, model, logger, charge_controller, discharge_controller):
        threading.Thread.__init__(self)
        self.delay = TIME_QUANT  # energy sensors updates at triple freq
        self.model = model
        self.run_event = run_event
        self.elastic = elastic
        self.logger = logger
        self.charge_controller = charge_controller
        self.discharge_controller = discharge_controller

    def run(self):
        i = 0
        battery_power = 0
        while self.run_event.is_set():
            try:
                self.model.process_cycle()
                wind_power = self.model.wind_gen.current_power
                print("Wind power = {}".format(wind_power))

                battery_power = self.model.battery.current_power

                gen_inverter_ref, load_inverter_ref = self.model.get_hardware_references()

                try:
                    send_to_elastic(self.elastic, self.model, wind_power, battery_power)
                except Exception as e:
                    stderr.write(str(e))

                print("send to battery:power = {}, energy = {} , soc = {}".format(battery_power,
                                                                                  self.model.battery.get_energy(),
                                                                                  self.model.battery.get_soc()))
                self.logger.log(BATTERY_LOG, [battery_power, self.model.battery.get_energy(), self.model.battery.get_soc()])
                modbus_simul_utils.write_battery_data(self.model.battery.get_soc(), self.model.battery.get_energy())

                data = model_utils.get_weather_and_states_data(self.model, wind_power)
                #print('send to others: ', data)
                self.logger.log(OTHERS_LOG, data)
                modbus_simul_utils.write_wind_data(wind_power)
                modbus_simul_utils.write_heaters_data(self.model)
                modbus_simul_utils.write_outside_temp_data(self.model.get_weather_at_time()[1])
                self.charge_controller.update(gen_inverter_ref / 1000)
                self.discharge_controller.update(- load_inverter_ref / 1000)
                i += 1
                time.sleep(self.delay)

            except ConnectionResetError as e:
                print(e)
                self.run_event.clear()


class ThreadListen(threading.Thread):
    def __init__(self, run_event, model, logger):
        threading.Thread.__init__(self)
        self.model = model
        self.run_event = run_event
        self.logger = logger

    def run(self):
        while self.run_event.is_set():
            for i in range(3):
                data = modbus_simul_utils.read_heater_data(i)
                with self.model.lock:
                    self.model.all_powers[i].append([time.time(), data['cfg_power']])
                    #self.model.save_state()
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='config/config.yml',
                        help='path to config file')

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    print(yaml.dump(config))
    random.seed(config['seed'])

    appenders = {}
    data_path = config['log_path']
    appenders[POWER_REFS_LOG] = os.path.join(data_path, 'prefs_log.csv')
    appenders[STATES_LOG] = os.path.join(data_path, 'states_log.csv')
    appenders[COMMON_LOG] = os.path.join(data_path, 'common_log')
    appenders[BATTERY_LOG] = os.path.join(data_path, 'battery_log.csv')
    appenders[OTHERS_LOG] = os.path.join(data_path, 'others_log.csv')
    logger = Logger(appenders)

    model = model_utils.get_model(TIME_QUANT, TIME_SCALE, logger, config['state_file'], config['weather_data'])

    try:
        model.load_state(config['state_file'])
    except FileNotFoundError:
        stderr.write('State file not found. Starting from scratch\n')

    run_event = threading.Event()
    run_event.set()

    elastic = Elasticsearch([f"http://{config['elasticsearch']['auth']}@{config['elasticsearch']['host']}"])

    charge_controller = battery_controller.create_controller(config['charge_host'], config['charge_port'], data_path,
                                                             "charge")
    discharge_controller = battery_controller.create_controller(config['discharge_host'], config['discharge_port'],
                                                                data_path, "discharge")
    send = ThreadSend(run_event, elastic, model, logger, charge_controller, discharge_controller)
    listen = ThreadListen(run_event, model, logger)

    if config['use_real_bat']:
        charge_controller.start()
        discharge_controller.start()

    send.start()
    listen.start()

    send.join()
    listen.join()


if __name__ == "__main__":
    main()
