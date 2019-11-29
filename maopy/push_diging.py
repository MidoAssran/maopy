"""
Push-DIGing for distributed optimization using column-stochastic mixing

:author: Mido Assran
:description: Distributed optimization using column stochastic mixing and
              gradient tracking. Based on the paper (nedich2016Achieving)
"""

import time

import numpy as np

from .gossip.gossip_comm import GossipComm
from .gossip.gossip_log import GossipLog
from .gossip.push_sum_gossip import PushSumAverager

# Message passing and network variables
COMM = GossipComm.comm
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name


class PushDIGing(object):
    """
    Distributed optimization using gradient tracking and column stochastic
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
        super(PushDIGing, self).__setattr__(name, value)

        if (name == 'peers') or (name == 'in_degree'):
            # Updating gossip averager property with $(name)
            try:
                psga = super(PushDIGing, self).__getattribute__('psga')
                setattr(psga, name, value)
            except AttributeError:
                # Do nothing if the instance hasn't been created yet
                pass

    def minimize(self):
        """
        Minimize objective using PushDIGing

        Procedure:
        1) Update: ps_argmin_n -= step_size * ps_grad_n
        2) Gossip: push_sum_gossip([ps_argmin_n, ps_grad_n, ps_w])
        3) Update: ps_result = push_sum_gossip([ps_argmin_n, ps_grad_n, ps_w])
            3.a) Update: ps_argmin_n = ps_result[ps_argmin_n]
            3.b) Update: ps_grad_n = ps_result[ps_grad_n]
            3.c) Update: ps_w = ps_result[ps_w]
            3.d) Update: argming_est = ps_argmin_n / ps_w
            3.e) Update: ps_grad_n += grad(argmin_est@(k)) - grad(argmin_est@(k-1))
        3) Repeat until completed $(termination_condition) itr.

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """

        # ----Initialize push diging---- #
        argmin_est = self.argmin_est
        ps_argmin_n = self.argmin_est
        ps_w = 1.0
        ps_grad_n_k = self.sub_gradient(argmin_est)
        grad_km1 = ps_grad_n_k

        # Setup loop parameters
        itr = 0
        if self.log:
            # -- log argmin-estimate
            l_argmin_est = GossipLog()
            l_argmin_est.log(argmin_est, itr)
            # -- log push-sum-weight
            l_ps_w = GossipLog()
            l_ps_w.log(ps_w, itr)

        # Start optimization at the same time
        COMM.Barrier()
        print('%s: starting optimization...' % UID)
        np.random.seed(UID)

        # Optimization loop
        while not self.the_terminator(itr):

            # --- update iteration
            itr += 1

            # -- take a step and gossip
            ps_argmin_n -= self.step_size * ps_grad_n_k
            gossip_vector = np.append(ps_argmin_n, ps_grad_n_k)
            ps_result = self.psga.gossip(gossip_vector, ps_w, asynch=False)
            print('%s: itr(%s) %s' % (UID, itr, ps_w))

            # -- update argmin estimate
            ps_w = ps_result['ps_w']
            ps_n = np.array(ps_result['ps_n'])
            ps_argmin_n = ps_n[:ps_argmin_n.size].reshape(ps_argmin_n.shape)
            argmin_est = ps_argmin_n / ps_w

            # -- update gradient tracking estimate
            grad_k = self.sub_gradient(argmin_est)
            ps_grad_n_k = ps_n[ps_argmin_n.size:].reshape(ps_grad_n_k.shape)
            ps_grad_n_k += grad_k - grad_km1
            # -- update the past gradient (at itr. k minus 1)
            grad_km1 = grad_k

            # -- log the varaibles
            if self.log:
                l_argmin_est.log(argmin_est, itr)
                l_ps_w.log(ps_w, itr)

        self.argmin_est = argmin_est

        if self.log:
            return {"argmin_est": l_argmin_est,
                    "ps_w": l_ps_w}
        else:
            return {"argmin_est": argmin_est,
                    "objective": self.objective(argmin_est),
                    "sub_gradient": self.gradient(argmin_est)}


if __name__ == "__main__":

    def demo(num_instances_per_node, num_features):
        """
        Demo fo the use of the GradientPush class.

        To run the demo, run the following from the multi_agent_optimization
        directory CLI:
            mpiexec -n $(num_nodes) python -m maopy.gradient_push
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

        pushdiging = PushDIGing(objective=objective,
                                sub_gradient=gradient,
                                arg_start=x_start,
                                peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                                step_size=1e-5,
                                max_itr=1000,
                                in_degree=2)

        loggers = pushdiging.minimize()
        l_argmin_est = loggers['argmin_est']

        print('%s: (start: %s)(finish: %s)'
              % (UID, obj_start, globjective(l_argmin_est.gossip_value)))

    # Run a demo where nodes minimize a sum of squares function
    demo(num_instances_per_node=5000, num_features=100)
