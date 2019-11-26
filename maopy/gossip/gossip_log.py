"""
Object storing iteration and time stamped history of a variable over an sequential process.

:author: Mido Assran
:description: Data structure for storing the time-stamped and iteration history of an incrementally
              updated variable.
"""

import time

import numpy as np


class GossipLog(object):
    """
    Data structure storing iteration- and time-stamped history of a variable.

    :param history: itr-stamped list made up of tuples (time (sec), value)
    :param start_time: Time (seconds) at which the logging process was started
    :param end_time: Time (seconds) at which the last variable was logged
    :param end_itr: Iteration at which the last variable was logged
    :param gossip_value: Most recently logged variable value
    :param log_freq: The iteration-freqeuncy for saving "logged" variables
    """

    def __init__(self, log_freq=100):
        self.history = {}
        self.start_time = time.time()
        self.end_time = None
        self.end_itr = None
        self.gossip_value = None
        self.log_freq = log_freq

    def log(self, value, itr, time_offset=0., force=False):
        """ Log the variable with an iteration and time stamp. """
        if (itr % self.log_freq != 0) and (not force):
            return
        tnow = time.time() - self.start_time + time_offset
        value = np.copy(value)
        self.history[itr] = (tnow, value)

        self.end_time = tnow
        self.end_itr = itr
        self.gossip_value = value
