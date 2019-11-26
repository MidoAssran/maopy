"""
EXTRAPush for distributed optimization using column stochastic mixing

:author: Mido Assran
:description: Distributed otpimization using column stochastic mixing and
              static gradient tracking. Based on the paper (zeng2015extra)
"""

import numpy as np

from .gossip.gossip_comm import GossipComm
from .gossip.gossip_log import GossipLog
from .gossip.push_sum_gossip import PushSumAverager

# Message passing and network variables
COMM = GossipComm.comm
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name


class ExtraPush(object):
    """
    Distributed optimization static gradient tracking and column stochastic
    mixing.
    """

    def __init__(self,
                 objective,
                 sub_gradient,
                 arg_start,
                 peers=[(UID+1) % SIZE],
                 step_size=1e-2,
                 max_itr=100,
                 in_degree=SIZE,
                 log=True):
        """ Initialize the gossip optimization settings. """

        self.objective = objective
        self.sub_gradient = sub_gradient
        self.argmin_est = np.array(arg_start)

        self.peers = peers
        self.step_size = step_size
        self.max_itr = max_itr

        # Termination function
        def terminator(itr=0):
            return (itr > self.max_itr)
        self.the_terminator = terminator

        self.in_degree = in_degree
        self.log = log

        self.psga = PushSumAverager(peers, in_degree)

    def __setattr__(self, name, value):
        super(ExtraPush, self).__setattr__(name, value)

        if (name == 'peers') or (name == 'in_degree'):
            # Updating gossip averager property with $(name)
            try:
                psga = super(ExtraPush, self).__getattribute__('psga')
                setattr(psga, name, value)
            except AttributeError:
                # Do nothing if the instance hasn't been created yet
                pass

    def minimize(self):
        """
        Minimize objective using ExtraPush

        Procedure:
        1) Gossip: push_sum_gossip([ps_n_km1, ps_w_km1])
        2) Update: ps_result = push_sum_gossip([ps_n_km1, ps_w_km1])
            2.a) Update: ps_n_k = ps_result[ps_n_km1] + ExtraPush step
            2.b) Update: ps_w_k = ps_result[ps_w_km]
            2.c) Update: argming_est = ps_n_k / ps_w_k
        3) Repeat until completed $(termination_condition) itr.

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """

        # --Initialize ExtraPush Parameters-- #

        # -- initialize (k minus 2) vlaues
        ps_n_km2 = self.argmin_est
        ps_w_km2 = 1.0
        argmin_est_km2 = ps_n_km2 / ps_w_km2
        grad_km2 = self.sub_gradient(argmin_est_km2)

        # -- gossip (k minus 2) values
        ps_result = self.psga.gossip(ps_n_km2, ps_w_km2, asynch=False)

        # -- initialize (k minus 1) values using gossip results;
        # (first time is just gradient descent)
        ps_w_km1 = ps_result['ps_w']
        p_ps_n_km2 = ps_result['ps_n']  # p_ps_.. = (post push sum)
        ps_n_km1 = p_ps_n_km2 - self.step_size * grad_km2
        argmin_est_km1 = ps_n_km1 / ps_w_km1
        grad_km1 = self.sub_gradient(argmin_est_km1)

        # -- setup loop parameters
        itr = 0
        if self.log:
            # -- log the argmin estimate (k minus 2)
            l_argmin_est = GossipLog()
            l_argmin_est.log(argmin_est_km2, itr)
            # -- log the push-sum weight
            l_ps_w = GossipLog()
            l_ps_w.log(ps_w_km2, itr)
            # -- log estimates at iteration (k minus 1)
            itr += 1
            l_argmin_est.log(argmin_est_km1, itr)
            l_ps_w.log(ps_w_km1, itr)

        # Start optimization at the same time
        COMM.Barrier()
        print('%s: starting optimization...' % UID)
        np.random.seed(UID)

        # Optimization loop
        while not self.the_terminator(itr):

            # -- update iteration
            itr += 1

            # -- gossip, take a step, and update argmin estimate
            ps_result = self.psga.gossip(ps_n_km1, ps_w_km1, asynch=False)
            ps_w_k = ps_result['ps_w']
            p_ps_n_km1 = ps_result['ps_n']
            ps_n_k = (p_ps_n_km1 + ps_n_km1) - 0.5 * (p_ps_n_km2 + ps_n_km2) \
                     - self.step_size * (grad_km1 - grad_km2)
            argmin_est = ps_n_k / ps_w_k
            print('%s: itr(%s) %s' % (UID, itr, ps_w_k))

            # -- update the iteration indices on the ExtraPush variables
            ps_n_km2 = ps_n_km1
            p_ps_n_km2 = p_ps_n_km1
            ps_n_km1 = ps_n_k
            ps_w_km1 = ps_w_k
            grad_km2 = grad_km1
            grad_km1 = self.sub_gradient(argmin_est)

            # -- log the varaibles
            if self.log:
                l_argmin_est.log(argmin_est, itr)
                l_ps_w.log(ps_w_k, itr)

        self.argmin_est = argmin_est

        if self.log:
            return {"argmin_est": l_argmin_est,
                    "ps_w": l_ps_w}
        else:
            return {"argmin_est": argmin_est,
                    "objective": objective(argmin_est),
                    "sub_gradient": gradient(argmin_est)}


if __name__ == "__main__":

    def demo(num_instances_per_node, num_features):
        """
        Demo fo the use of the GradientPush class.

        To run the demo, run the following from the maopy directory CLI:
            mpiexec -n $(num_nodes) python -m maopy.exra_push
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

        extrapush = ExtraPush(objective=objective,
                              sub_gradient=gradient,
                              arg_start=x_start,
                              peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                              step_size=1e-5,
                              max_itr=1000,
                              in_degree=2)

        loggers = extrapush.minimize()
        l_argmin_est = loggers['argmin_est']

        print('%s: (start: %s)(finish: %s)'
              % (UID, obj_start, globjective(l_argmin_est.gossip_value)))

    # Run a demo where nodes minimize a sum of squares function
    demo(num_instances_per_node=5000, num_features=100)
