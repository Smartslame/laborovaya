import subprocess


def cdaput(device, param, value):
    args = ["cdaput", "-g", device, device + param, " {}".format(value)]
    subprocess.Popen(args, stdout=subprocess.PIPE)


def read_heater_data(i):
    device = "heater{}".format(i + 1)
    bash_command = "cdacat -g " + device
    process = subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    return {line.split('\t')[0].lstrip(device): float(line.split('\t')[1]) for line in
            output.strip().decode('utf-8').split('\n')}


def write_battery_data(power, energy):
    device = "battery"
    cdaput(device, "status_power", power)
    cdaput(device, "status_energy", energy)


def write_wind_data(power):
    device = "wind"
    print("write data to device " + device)
    print("status_power " + str(power))
    cdaput(device, "status_power", power)


def write_outside_temp_data(temp):
    # TODO reconfig modbus simul
    device = "heater{}".format(4)
    print("write data to device " + device)
    print("outdoor_temperature " + str(temp))
    cdaput(device, "indoor_temperature", temp)


def write_heaters_data(model):
    for i in range(model.num_buildings):
        device = "heater{}".format(i + 1)
        print("write data to device " + device)
        print("status_power " + str(model.buildings[i].current_power))
        print("indoor_temperature " + str(model.buildings[i].get_temp()))
        cdaput(device, "status_power", model.buildings[i].current_power)
        cdaput(device, "indoor_temperature", model.buildings[i].get_temp())
