"""
ASY-SONATA class for parallel optimization using row-/column-stochastic mixing.

:author: Mido Assran
:description: Distributed optimization using row- and column-stochastic mixing
              and gradient tracking. Based on the paper (tian2019Achieving)
"""

import time

import numpy as np

from .gossip_comm import GossipComm
from .robust_push_gossip_averaging import RobustPushAverager
from .pull_gossip_averaging import PullGossipAverager

# Message passing and network variables
COMM = GossipComm.comm
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name

DEFAULT_STEP_SIZE = 1e-2


class AsySONATA(object):
    """
    Distributed asynchronous optimization with gradient racking using row-
    and column-stochastic mixing.
    """

    def __init__(self, objective,
                 sub_gradient,
                 arg_start,
                 peers=None,
                 step_size=None,
                 termination_condition=None,
                 in_degree=SIZE,
                 log=True):
        """ Initialize the gossip optimization settings. """

        self.argmin_est = np.array(arg_start)
        self.objective = objective
        self.sub_gradient = sub_gradient
        if peers is None:
            peers = [i for i in range(SIZE) if i != UID]
        self.peers = peers

        if step_size is None:
            step_size = DEFAULT_STEP_SIZE
        self.step_size = step_size

        self.termination_condition = termination_condition
        self.in_degree = in_degree
        self.log = log

        self.rpga = RobustPushAverager(peers, in_degree)
        self.plga = PullGossipAverager(peers, in_degree)

    def __setattr__(self, name, value):
        super(AsySONATA, self).__setattr__(name, value)

        if (name == 'peers') or (name == 'in_degree'):
            # Updating gossip averager property with $(name)
            try:
                rpga = super(AsySONATA, self).__getattribute__('rpga')
                setattr(rpga, name, value)
                plga = super(AsySONATA, self).__getattribute__('plga')
                setattr(plga, name, value)
            except AttributeError:
                # Do nothing if the instance hasn't been created yet
                pass

    def minimize(self):
        """
        Minimize the objective specified in settings using the PushDIGing procedure

        Procedure:
        1) Update: argmin_est -= step_size * ps_grad_n
        2) Gossip: argmin_est = pull_gossip([argmin_est])
        3) Update: ps_grad_n += sub_gradient(argmin_est@(k)) - sub_gradient(argmin_est@(k-1))
        4) Gossip: ps_grad_n = robust_push_gossip(ps_grad_n)
        3) Repeat until completed $(termination_condition) time

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """

        rpga = self.rpga
        plga = self.plga

        step_size = self.step_size
        gradient = self.sub_gradient
        objective = self.objective

        # ----Initialize asy-sonata---- #
        argmin_est = self.argmin_est
        ps_grad_n_k = gradient(argmin_est)
        grad_km1 = ps_grad_n_k

        # Setup loop parameters
        itr = 0
        log = self.log
        if log:
            # log the argmin estimate
            from .gossip_log import GossipLog
            l_argmin_est = GossipLog()
            l_argmin_est.log(argmin_est, itr)

        end_time = time.time() + self.termination_condition
        condition = time.time() < end_time

        # Start optimization at the same time
        COMM.Barrier()
        print('%s: starting optimization...' % UID)
        np.random.seed(UID)

        # Optimization loop
        while condition:

            # Update iteration
            itr += 1

            # -- START AsySONATA update -- #

            # -- local descent step
            argmin_est -= step_size * ps_grad_n_k
            # -- pull (row-stochastic) gossip argmin-est
            argmin_est = plga.gossip(gossip_value=argmin_est)

            # -- update gradient tracking estimate
            grad_k = gradient(argmin_est)
            ps_grad_n_k += grad_k - grad_km1
            grad_km1 = grad_k
            # -- robust-push-sum gossip gradient tracker (discard ps-weight)
            # ps_grad_n_k = rpga.gossip(gossip_value=ps_grad_n_k)

            # -- END AsySONATA update -- #

            # Log the varaibles
            if log:
                l_argmin_est.log(argmin_est, itr)

            # Update the termination flag
            condition = time.time() < end_time

        COMM.Barrier()
        # Fetch any lingering message
        print('%s: Fetching lingering msgs...' % (UID))
        timeout = time.time()
        while (time.time() - timeout) <= 5.:
            argmin_est = plga.gossip(gossip_value=argmin_est,
                                     just_probe=True)
            ps_grad_n_k = rpga.gossip(gossip_value=ps_grad_n_k,
                                      just_probe=True)
            time.sleep(0.5)
        barrier_time = (time.time() - timeout)
        # Log the varaibles
        if log:
            itr += 1
            l_argmin_est.log(argmin_est, itr,
                             time_offset=(-barrier_time))

        self.argmin_est = argmin_est

        if log is True:
            return {"argmin_est": l_argmin_est}
        else:
            return {"argmin_est": argmin_est,
                    "objective": objective(argmin_est),
                    "sub_gradient": gradient(argmin_est)}


if __name__ == "__main__":

    def demo(num_instances_per_node, num_features):
        """
        Demo fo the use of the AsySONATA class.

        To run the demo, run the following from the multi_agent_optimization directory CLI:
            mpiexec -n $(num_nodes) python -m maopy.asy_sonata
        """
        # Starting point
        np.random.seed(seed=UID)
        x_start = np.random.randn(num_features, 1)

        # Create objective function and its gradient
        np.random.seed(seed=0)
        ga_m = np.random.randn(SIZE * num_instances_per_node, num_features)
        gb_v = np.random.randn(SIZE * num_instances_per_node, 1)
        globjective = lambda x: 0.5 * (np.linalg.norm(ga_m.dot(x) - gb_v))**2
        obj_start = globjective(x_start)

        # Partition objective function into local objectives
        start = UID * num_instances_per_node
        a_m = ga_m[start:start+num_instances_per_node]
        b_v = gb_v[start:start+num_instances_per_node]
        objective = lambda x: 0.5 * (np.linalg.norm(a_m.dot(x) - b_v))**2
        gradient = lambda x: a_m.T.dot(a_m.dot(x)-b_v)

        asysonata = AsySONATA(objective=objective,
                              sub_gradient=gradient,
                              arg_start=x_start,
                              peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                              step_size=1e-5,
                              termination_condition=.5,
                              in_degree=2)

        loggers = asysonata.minimize()
        l_argmin_est = loggers['argmin_est']

        print('%s: (start: %s)(finish: %s)'
              % (UID, obj_start, globjective(l_argmin_est.gossip_value)))

    # Run a demo where nodes minimize a sum of squares function
    demo(num_instances_per_node=5000, num_features=100)
