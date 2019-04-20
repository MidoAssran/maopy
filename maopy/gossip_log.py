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
    Data structure class storing iteration and time stamped history of a variable.

    :param history: Iteration stamped variable history storing the tuple (time (sec), value)
    :param start_time: Time (seconds) at which the logging process was started
    :param end_time: Time (seconds) at which the last variable was logged
    :param end_itr: Iteration at which the last variable was logged
    :param gossip_value: Most recently logged variable value
    """

    def __init__(self):
        self.history = {}
        self.start_time = time.time()
        self.end_time = None
        self.end_itr = None
        self.gossip_value = None

    def log(self, value, itr):
        """ Log the variable with an iteration and time stamp. """
        tnow = time.time() - self.start_time
        value = np.copy(value)
        self.history[itr] = (tnow, value)

        self.end_time = tnow
        self.end_itr = itr
        self.gossip_value = value
