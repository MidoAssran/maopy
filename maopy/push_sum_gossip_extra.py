"""
Push Sum Gossip EXTRA class for parallel optimization using column stochastic mixing.

:author: Mido Assran
:description: Distributed otpimization using column stochastic mixing and static gradient tracking.
              Based on the paper (zeng2015extra)
"""

import time

import numpy as np

from .gossip_comm import GossipComm
from .push_sum_optimization import PushSumOptimizer

# Message passing and network variables
COMM = GossipComm.comm
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name


class ExtraPush(PushSumOptimizer):
    """ Distributed optimization static gradient tracking and column stochastic mixing. """

    # Inherit docstring
    __doc__ += PushSumOptimizer.__doc__

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

        super(ExtraPush, self).__init__(objective=objective,
                                        sub_gradient=sub_gradient,
                                        arg_start=arg_start,
                                        synch=synch,
                                        peers=peers,
                                        step_size=step_size,
                                        terminate_by_time=terminate_by_time,
                                        termination_condition=termination_condition,
                                        log=log,
                                        out_degree=out_degree,
                                        in_degree=in_degree,
                                        num_averaging_itr=num_averaging_itr,
                                        all_reduce=all_reduce)


    def minimize(self):
        """
        Minimize the objective specified in settings using the PushDIGing procedure

        Procedure:
        1) Gossip: push_sum_gossip([ps_n_km1, ps_w_km1])
        2) Update: ps_result = push_sum_gossip([ps_n_km1, ps_w_km1])
            2.a) Update: ps_n_k = ps_result[ps_n_km1] + ExtraPush step
            2.b) Update: ps_w_k = ps_result[ps_w_km]
            2.c) Update: argming_est = ps_n_k / ps_w_k
        3) Repeat until completed $(termination_condition) itr. or time (depending on settings)

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """
        super(ExtraPush, self).minimize()

        psga = self.ps_averager

        step_size = self.step_size
        gradient = self.sub_gradient
        objective = self.objective


        # Start optimization at the same time
        COMM.Barrier()


        # -- Initialize ExtraPush Parameters -- #

        # Initialize (k minus 2) vlaues
        ps_n_km2 = self.argmin_est
        ps_w_km2 = 1.0
        argmin_est_km2 = ps_n_km2 / ps_w_km2
        grad_km2 = gradient(argmin_est_km2)

        # Gossip (k minus 2) values
        ps_result = psga.gossip(gossip_value=ps_n_km2, ps_weight=ps_w_km2)

        # Initialize (k minus 1) values using gossip results (first time is just gradient descent)
        ps_w_km1 = ps_result['ps_w']
        p_ps_n_km2 = ps_result['ps_n'] # p_ps_.. = (post push sum)
        ps_n_km1 = p_ps_n_km2 - step_size * grad_km2
        argmin_est_km1 = ps_n_km1 / ps_w_km1
        grad_km1 = gradient(argmin_est_km1)

        # -- END Initialize ExtraPush Parameters -- #

        # Setup loop parameters
        itr = 0
        log = self.log
        if log:
            from .gossip_log import GossipLog
            l_argmin_est = GossipLog() # Log the argmin estimate
            l_ps_w = GossipLog() # Log the push sum weight
            l_argmin_est.log(argmin_est_km2, itr)
            l_ps_w.log(ps_w_km2, itr)
            itr += 1
            l_argmin_est.log(argmin_est_km1, itr)
            l_ps_w.log(ps_w_km1, itr)

        # Setup our running condition (time or iteration)
        if self.terminate_by_time is False:
            num_gossip_itr = self.termination_condition
            condition = itr < num_gossip_itr
        else:
            end_time = time.time() + self.termination_condition
            condition = time.time() < end_time

        # Optimization loop
        while condition:

            if self.synch is True:
                COMM.Barrier()

            # Update iteration
            itr += 1

            # -- START ExtraPush update -- #

            # Gossip, take a step, and update argmin estimate
            ps_result = psga.gossip(gossip_value=ps_n_km1, ps_weight=ps_w_km1)
            ps_w_k = ps_result['ps_w']
            p_ps_n_km1 = ps_result['ps_n']
            ps_n_k = (p_ps_n_km1 + ps_n_km1) - 0.5 * (p_ps_n_km2 + ps_n_km2) \
                     - step_size * (grad_km1 - grad_km2)
            argmin_est = ps_n_k / ps_w_k
            print('%s: itr(%s) %s' % (UID, itr, ps_w_k))

            # Update the iteration indices on the ExtraPush variables
            ps_n_km2 = ps_n_km1
            p_ps_n_km2 = p_ps_n_km1
            ps_n_km1 = ps_n_k
            ps_w_km1 = ps_w_k
            grad_km2 = grad_km1
            grad_km1 = gradient(argmin_est)

            # -- END ExtraPush update -- #

            # Log the varaibles
            if log:
                l_argmin_est.log(argmin_est, itr)
                l_ps_w.log(ps_w_k, itr)


            # Update the running condition
            if self.terminate_by_time is False:
                condition = itr < num_gossip_itr
            else:
                condition = time.time() < end_time

        self.argmin_est = argmin_est

        if log is True:
            return {"argmin_est": l_argmin_est,
                    "ps_w": l_ps_w}
        else:
            return {"argmin_est": argmin_est,
                    "objective": objective(argmin_est),
                    "sub_gradient": gradient(argmin_est)}


if __name__ == "__main__":

    def demo(num_instances_per_node, num_features):
        """
        Demo fo the use of the PushSumSubgradientDescent class.

        To run the demo, run the following from the multi_agent_optimization directory CLI:
            mpiexec -n $(num_nodes) python -m do4py.push_sum_gossip_gradient_descent
        """

        # Create objective function and its gradient
        np.random.seed(seed=UID)
        x_start = np.random.randn(num_features, 1)
        a_m = np.random.randn(num_instances_per_node, num_features)
        b_v = np.random.randn(num_instances_per_node, 1)
        objective = lambda x: 0.5 * (np.linalg.norm(a_m.dot(x) - b_v))**2
        gradient = lambda x: a_m.T.dot(a_m.dot(x)-b_v)

        ep = ExtraPush(objective=objective,
                       sub_gradient=gradient,
                       arg_start=x_start,
                       synch=True,
                       peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                       step_size=1e-4,
                       terminate_by_time=False,
                       termination_condition=1000,
                       log=True,
                       in_degree=2,
                       num_averaging_itr=1)

        loggers = ep.minimize()
        l_argmin_est = loggers['argmin_est']

        l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=True)

    # Run a demo where nodes minimize a sum of squares function
    demo(num_instances_per_node=5000, num_features=100)
