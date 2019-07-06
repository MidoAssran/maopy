"""
Push Sum Optimization composite design pattern abstract class.

:author: Mido Assran
:description: Abstract class to be used in a composite design pattern unifying the structure of the
              various push sum based optimization methods.
"""

import warnings

import numpy as np

from .gossip_comm import GossipComm
from .push_sum_gossip_averaging import PushSumGossipAverager as PSGA

# Message passing and network variables
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name

# Default values
DEFAULT_NUM_GOSSIP_ITR = 1
DEFAULT_GOSSIP_TIME = 0.1 # time in seconds
DEFAULT_STEP_SIZE = 1e-2

class PushSumOptimizer(object):
    """
    Abstract class for composite design pattern architecture of gossip optimization methods.

    :param argmin_est: Estimate of the argument that minimizes the sum of nodes' objectives
    :param objective: Local objective lambda
    :param sub_gradient: Local subgradient lambda (MUST return numpy arr. even if scalar valued)
    :param synch: Whether to run the alg. synchronusly (or asynchronously)
    :param peers: UniqueIDs of neighbouring peers in net. (used for comm.)
    :param step_size: The starting step-size of the algorithm
    :param terminate_by_time: Whether to terminate the alg. after some threshold time
    :param termination_condition: Itr. count by default, otherwise threshold time
    :param log: Whether to log the alg. variables at each iteration
    :param out_degree: Num. of rand. peers to choose/communicate with at each itr. (all by default)
    :param in_degree: Num. messages to expect at each itr. (only used in static synchronous nets.)
    :param ps_averager: Instance of consensus averaging module used for inter-iteration gossip
    :param num_averaging_itr: Num. averaging itr. to perform each optimization round
    :param all_reduce: Whether to perform MPI All Reduce for averaging rather than gossip.
    """

    def __init__(self, objective,
                 sub_gradient,
                 arg_start,
                 synch=True,
                 peers=None,
                 step_size=None,
                 terminate_by_time=False,
                 termination_condition=None,
                 log=False,
                 out_degree=None,
                 in_degree=SIZE,
                 num_averaging_itr=1,
                 all_reduce=False):
        """ Initialize the gossip optimization settings. """

        self.argmin_est = np.array(arg_start)

        self.objective = objective

        self.sub_gradient = sub_gradient

        self.synch = synch

        if peers is None:
            peers = [i for i in range(SIZE) if i != UID]
        self.peers = peers

        if step_size is None:
            step_size = DEFAULT_STEP_SIZE
        self.step_size = step_size

        self.terminate_by_time = terminate_by_time

        # Set the termination condition to the class defaults if not specified
        if termination_condition is None:
            if terminate_by_time:
                self.termination_condition = DEFAULT_GOSSIP_TIME
            else:
                self.termination_condition = DEFAULT_NUM_GOSSIP_ITR
        else:
            self.termination_condition = termination_condition

        self.log = log

        if out_degree is None:
            self.out_degree = len(self.peers)
        else:
            self.out_degree = out_degree

        self.in_degree = in_degree

        self.num_averaging_itr = num_averaging_itr

        self.all_reduce = all_reduce

        self.ps_averager = PSGA(synch=self.synch,
                                peers=self.peers,
                                terminate_by_time=False,
                                termination_condition=self.num_averaging_itr,
                                log=False,
                                out_degree=self.out_degree,
                                in_degree=self.in_degree,
                                all_reduce=self.all_reduce)

        if (self.synch is True) and (self.terminate_by_time is True):
            warnings.warn("Use of synchronous gossip w/ time term. cond. will result in deadlocks.")


    def __setattr__(self, name, value):
        super(PushSumOptimizer, self).__setattr__(name, value)

        if (name == 'synch') or \
           (name == 'peers') or \
           (name == 'num_averaging_itr') or \
           (name == 'out_degree') or \
           (name == 'in_degree') or \
           (name == 'all_reduce'):
            # Updating gossip averager property with $(name)
            try:
                ps_averager = super(PushSumOptimizer, self).__getattribute__('ps_averager')
                setattr(ps_averager, name, value)
            except AttributeError:
                # Do nothing if the instance hasn't been created yet
                pass


    def minimize(self):
        """ The optimization procedure to be implemented by the inheriting class. """

        if (self.synch is True) and (self.terminate_by_time is True):
            warnings.warn("Use of synchronous gossip w/ time term. cond. will result in deadlocks.")






