import numpy as np
import time


class Runtime():
    '''
    record the running time during training
    '''

    def __init__(self):
        self.time_epoch_list = []
        self.start_time = time.time()

    def epoch_start(self):
        self.i_start_time = time.time()

    def epoch_end(self):
        elapsed_times_epoch = time.time() - self.i_start_time
        self.time_epoch_list.append(elapsed_times_epoch)

    def sum_elapsed_time(self):
        return np.sum(self.time_epoch_list)

    def mean_elasped_time(self):
        return np.mean(self.time_epoch_list)

    def std_elasped_time(self):
        return np.std(self.time_epoch_list)

    def print_mean_sum_time(self, prefix=''):
        print(
            f"{prefix} RunningTime: mean={self.mean_elasped_time():.5f}$\pm${self.std_elasped_time():.5f} || sum={self.sum_elapsed_time():.5f}")
