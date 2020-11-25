import random
import time
from datetime import datetime, timedelta
from sys import stderr

import numpy as np
import pandas as pd

from model import Building, Model, WindGen, Battery


def get_weather_data(TIME_QUANT, TIME_SCALE, nrows=None, weather_data_path=None):
    # nrows = 1000
    print('loading data')
    dfi = pd.read_csv(weather_data_path, sep=',', nrows=nrows)
    dfi.time = dfi.time.apply(lambda s: datetime.strptime(s, '%d.%m.%Y %H:%M:%S'))
    # проверка на непрерывность времени
    if not np.all(((dfi.time - np.roll(dfi.time, 1))[1:] == timedelta(0, TIME_QUANT * TIME_SCALE))):
        stderr.write('Time continuosity violated at indices: ' +
                     str(np.where((dfi.time - np.roll(dfi.time, 1))[1:] != timedelta(0, TIME_QUANT * TIME_SCALE)))
                     + '\n')
        # if input('continue? y/n:') != 'y':
        #     return
    # dfi['P'] = dfi.v.apply(gen_power)
    # dfi = dfi[:3700002] # есть пропуски дальше
    # dfi['P'] = dfi.v.apply(gen_power)
    print('data loaded')
    return dfi


def create_buildings(TIME_QUANT):
    r = lambda: 0.8 + random.random() * 0.4
    buildings = [Building(size=(8 * r() * (max_power / 20000), 10 * r() * (max_power / 20000), 2.5 * r()),
                 beta=15 * r(), R_st=0.5 * r(), sigma=0.03 * r(), max_power=max_power,
                 description='wooden house', time_quant=TIME_QUANT) 
                for max_power in [6000, 12000, 18000]]

    print('sum of nominal heaters powers (losses at -30C), kW: ',
          sum([b.get_loss_P() * (20 - (-30)) for b in buildings]) / 1000)
    return buildings


def get_weather_and_states_data(model, wind_power):
    wind, T_out = model.get_weather_at_time()
    print('weather: ', wind, T_out)

    data = np.zeros(3 + 2 * model.num_buildings, dtype=np.float32)

    data[0] = wind
    data[1] = T_out
    data[2] = wind_power

    for i in range(model.num_buildings):
        data[3 + i] = model.buildings[i].current_power
        data[3 + model.num_buildings + i] = model.buildings[i].get_temp()

    return data


def get_model(time_quant, time_scale, logger, state_file, weather_data_path, nrows=None):
    dfi = get_weather_data(time_quant, time_scale, nrows=nrows, weather_data_path=weather_data_path)
    return Model(create_buildings(time_quant), WindGen(), Battery(30000, min_soc=0.2, max_soc=0.8), dfi, time.time(), logger, state_file,
                 time_scale)
