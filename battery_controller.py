import logging
import socket
import struct
import threading
import time


class WorkingThread(threading.Thread):
    def __init__(self, run_event, soc, data, min_power, max_power, logger):
        threading.Thread.__init__(self)
        self.lock = threading.Lock()
        self.run_event = run_event
        self.soc = soc
        self.data = data
        self.min_power = min_power
        self.max_power = max_power
        self.cur_power = 0.

        self.default_command = float(1)
        self.default_power = float(0)
        self.time_tic = 0.2
        self.logger = logger

    def update(self, power):
        self.lock.acquire()
        self.cur_power = power
        self.lock.release()

    def run(self):
        while self.run_event.is_set():
            self.lock.acquire()
            packet = [self.default_command, self.cur_power]
            self.lock.release()
            packet = bytearray(struct.pack('>{0}f'.format(len(packet)), *packet))
            time.sleep(self.time_tic)
            self.logger.info("Send command with power: {}".format(self.cur_power))
            self.soc.send(packet)


def setup_logger(folder, name):
    handler = logging.FileHandler(folder + "/" + name + ".log")
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d  %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def create_controller(host, port, folder, name, min_power=-25., max_power=25.):
    host = host
    port = int(port)
    min_power = float(min_power)
    max_power = float(max_power)
    logger = setup_logger(folder, name)
    soc = socket.socket(socket.AF_INET,  # Internet
                            socket.SOCK_DGRAM)  # UDP
    soc.connect((host, port))

    run_event = threading.Event()

    run_event.set()

    return WorkingThread(run_event, soc, [], min_power, max_power, logger)
