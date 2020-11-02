import math
import random
import threading
import time

import numpy as np
from scipy.interpolate import interp1d


class WindGen:
    def __init__(self):
        v_tab = np.hstack((2.5, np.arange(3, 12)))
        p_tab = np.array([0, 1.5, 6, 13, 22, 35, 50, 70, 88, 100]) * 1000
        self.p_interp = interp1d(v_tab, p_tab, kind='quadratic')
        self.energy = 0

    def gen_power(self, v):
        return (v >= 2.5) * self.p_interp(np.clip(v, 2.5, 11))

    def update(self, v, time_delta):
        power = self.gen_power(v)
        self.energy += power * time_delta
        return power

    def get_energy(self):
        return self.energy / 3600


class Battery:
    def __init__(self, capacity, min_soc=0.0, max_soc=1, initial_soc=0.5):
        """ capacity in Wh"""
        self.energy = capacity * 3600 * initial_soc
        self.capacity = capacity * 3600
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.max_power = math.inf
        self.min_power = -math.inf
        self.eta = 1

    def update(self, gen_power, load_powers, time_delta):
        power = (gen_power - load_powers.sum())
        self.energy += power * time_delta
        return power

    def get_soc(self):
        return self.energy / self.capacity

    def get_energy(self):
        return self.energy / 3600

    def energy_remains_to_full(self):
        return self.capacity * self.max_soc - self.energy

    def energy_remains(self):
        return max(self.energy, self.capacity * self.min_soc)


class Building:
    def __init__(self, time_quant, T=24, size=(15, 40, 2.5), R_st=0.3, beta=30, sigma=0.05,
                 description='теплоаккумулятор'):
        self.beta = beta * 3600  # коэффициент аккумуляции (сек), см. выше
        self.T = T
        self.time_quant = time_quant  # секунды
        a, b, h = size  # размеры здания
        self.A_nar = 2 * (h * a + h * b) + a * b  # наружная площадь, пол 1 этажа утеплён лучше
        self.A_0 = a * b  # площадь помещения
        self.alpha = 1 / R_st  # средний коэф. теплопотерь стен, полов, потолков и окон (м2 * К / Вт)
        self.C = self.beta * (self.A_nar * self.alpha)
        self.sigma = sigma  # случайные изменения теплопотерь, связанные с открытием форточек, выше у маленьких домов
        self.ventilation = 0  # моделируется случайным блужданием, чтобы не было частых резких скачков
        self.energy = 0  # показания счётчика
        self.description = description
        self.max_power = 20000  # round(self.get_loss_P() * (20 - (-40)) * 1.8)
        print(self.description, 'heat capacitance, kJ/K:', round(self.C / 1000), 'Max loss (-30C), kW:',
              round(self.get_loss_P() * (20 - (-30))) / 1000)
        self.current_power = 0

        print('size', size, 'beta', beta, 'R_st', R_st, 'T', T, 'sigma', sigma)

    def update_temp(self, P, T_out, time_delta=None):
        if not time_delta:
            time_delta = self.time_quant

        P = np.clip(P, 0, self.max_power)
        self.T += ((P / self.C - (self.T - T_out) / self.beta * (1 + np.clip(self.ventilation, 0, 5 * self.sigma)))
                   * time_delta)
        self.ventilation += (random.random() - 0.5) * self.sigma
        self.energy += P * time_delta
        self.current_power = P
        return P

    def get_temp(self):
        return self.T

    def get_energy(self):
        return self.energy / 3600

    def get_loss_P(self):
        return self.C / self.beta

    def get_state(self):
        return self.T, self.energy, self.ventilation

    def load_state(self, state):
        if not state:
            pass
        self.T, self.energy, self.ventilation = state


class Model:
    def __init__(self, buildings, wind_gen, battery, dfi, start_time, logger, state_file, time_scale):
        self.lock = threading.Lock()  # mutex
        self.buildings = buildings
        self.num_buildings = len(buildings)
        self.wind_gen = wind_gen
        self.battery = battery
        self.last_update = start_time
        self.dfi = dfi
        self.time_passed = 0  # experiment running in real time except time between restarts
        self.wind_unused_energy = 0
        self.diesel_energy = 0
        self.time_scale = time_scale
        self.current_wind_power = 0
        self.current_battery_power = 0
        self.balanser = 0

        # power reference history in pairs [time of power update, power]
        self.all_powers = [[[start_time, 0]] for _ in range(self.num_buildings)]
        self._logger = logger
        self._state_file = state_file

    def get_weather_at_time(self):
        time_quant = (self.dfi.time[1] - self.dfi.time[0]).seconds
        n = int(self.time_passed * self.time_scale / time_quant)
        n += 52105  # start from second year, the first is used for NN learning
        n %= self.dfi.shape[0]
        print('get row number {0}'.format(n))
        return self.dfi['v'][n], self.dfi['T'][n]

    def process_cycle(self):
        with self.lock:
            now = time.time()
            self.time_passed += now - self.last_update
            time_delta = (now - self.last_update) * self.time_scale
            avg_powers = np.zeros(self.num_buildings)
            wind, T_out = self.get_weather_at_time()

            if now != self.last_update:  # вроде иначе невозможно, но
                for n in range(self.num_buildings):
                    self.all_powers[n].append(
                        [now, self.all_powers[n][-1][1]])  # вставим правую границу цикла
                    times = np.array(self.all_powers[n])[:, 0]
                    powers = np.array(self.all_powers[n])[:, 1]
                    # print('times:', times - self.start_time)
                    # print('powers:', powers)
                    avg_powers[n] = ((np.roll(times, -1) - times)[:-1] * powers[:-1]).sum() / (now - self.last_update)
                    avg_powers[n] = self.buildings[n].update_temp(avg_powers[n], T_out, time_delta)
                    self.all_powers[n] = [self.all_powers[n][-1]]  # стираем прошедший цикл

        wind_power = self.wind_gen.update(wind, time_delta)

        if wind_power >= avg_powers.sum():
            bat_charge = wind_power - avg_powers.sum()
            if bat_charge * time_delta > self.battery.energy_remains_to_full():
                bat_charge = self.battery.energy_remains_to_full() / time_delta
            balancer = wind_power - avg_powers.sum() - bat_charge
            self.wind_unused_energy = balancer * time_delta
            self.diesel_energy = 0
        else:
            bat_discharge = avg_powers.sum() - wind_power
            if bat_discharge * time_delta > self.battery.energy_remains():
                bat_discharge = self.battery.energy_remains() / time_delta
            balancer = wind_power - avg_powers.sum() + bat_discharge
            self.diesel_energy = -balancer * time_delta
            self.wind_unused_energy = 0

        battery_power = self.battery.update(wind_power - balancer, avg_powers, time_delta)

        self.current_wind_power, self.current_battery_power, self.balancer = wind_power, battery_power, balancer
    
        self.last_update = now


    def get_hardware_references(self):
        """ Prepare data for real-life simualtion considering hardware capabilities """ 
        scale = 0.25
        return np.array([self.current_wind_power - self.balancer, sum([h.current_power for h in self.buildings])]) * scale

    def save_state(self):
        with open(self._state_file, 'w') as state_file:
            state_file.write(' '.join([str(x) for x in (self.time_passed, self.wind_gen.energy,
                                                        self.diesel_energy, self.battery.energy)]) + '\n')
            for b in self.buildings:
                state_file.write(' '.join([str(x) for x in b.get_state()]) + '\n')

    def load_state(self, fname):
        with open(fname, 'r') as state_file:
            self.time_passed, self.wind_gen.energy, self.diesel_energy, self.battery.energy = \
                [float(x) for x in state_file.readline().split(' ')]

            for b in self.buildings:
                b.load_state([float(x) for x in state_file.readline().split(' ')])
