"""
Push Sum Gossip Gradient Descent class for parallel optimization using column stochastic mixing.

:author: Mido Assran
:description: Distributed otpimization using column stochastic mixing and greedy gradient descent.
              Based on the paper (nedich2015distributed)
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

# Default constants
DEFAULT_LEARNING_RATE = 0.1 # Time in seconds


class PushSumSubgradientDescent(PushSumOptimizer):
    """ Distributed optimization using column stochastic mixing and greedy gradient descent. """

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
                 constant_step_size=False,
                 learning_rate=None,
                 all_reduce=False):
        """ Initialize the gossip optimization settings. """

        self.constant_step_size = constant_step_size
        if learning_rate is None:
            learning_rate = DEFAULT_LEARNING_RATE
        self.learning_rate = learning_rate

        super(PushSumSubgradientDescent, self).__init__(objective=objective,
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

    def _gradient_descent_step(self, ps_n, argmin_est, itr=None, start_time=None):
        """ Take step in direction of negative gradient, and return the new domain point. """

        # Diminshing step-size: 1 / sqrt(k)
        if self.constant_step_size is False:
            if self.synch is True:
                if itr is None:
                    raise ValueError("'itr' is NONE for synch.alg. w/ diminishing stepsize")
                effective_itr = itr
            else:
                if start_time is None:
                    raise ValueError("'start_time' is NONE for asynch.alg. w/ diminishing stepsize")
                effective_itr = int((time.time() - start_time) / self.learning_rate) + 1
        else:
            effective_itr = 1

        step_size = self.step_size / (effective_itr ** 0.5)

        return ps_n - (step_size * self.sub_gradient(argmin_est))


    def minimize(self):
        """
        Minimize the objective specified in settings using the Subgradient-Push procedure

        Procedure:
        1) Gossip: push_sum_gossip([ps_n, ps_w])
        2) Update: ps_result = push_sum_gossip([ps_n, ps_w])
            2.a) ps_n = ps_result[ps_n]
            2.b) ps_w = ps_result[ps_w]
            2.c) argmin_est = ps_n / ps_w
            2.d) ps_n = ps_n - step_size * sub_gradient(argmin_est)
        3) Repeat until completed $(termination_condition) itr. or time (depending on settings)

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """
        super(PushSumSubgradientDescent, self).minimize()

        # Initialize sub-gradient descent push sum gossip
        ps_n = self.argmin_est
        ps_w = 1.0
        argmin_est = ps_n / ps_w

        itr = 0

        log = self.log
        psga = self.ps_averager
        objective = self.objective
        gradient = self.sub_gradient

        if log:
            from .gossip_log import GossipLog
            l_argmin_est = GossipLog() # Log the argmin estimate
            l_ps_w = GossipLog() # Log the push sum weight
            l_argmin_est.log(argmin_est, itr)
            l_ps_w.log(ps_w, itr)


        if self.terminate_by_time is False:
            num_gossip_itr = self.termination_condition
            condition = itr < num_gossip_itr
        else:
            gossip_time = self.termination_condition
            end_time = time.time() + gossip_time # End time of optimization
            condition = time.time() < end_time

        # Goes high if a message was not received in the last gossip round
        just_probe = False

        # Start optimization at the same time
        COMM.Barrier()

        start_time = time.time()

        # Optimization loop
        while condition:

            if self.synch is True:
                COMM.Barrier()

            itr += 1

            # -- START Subgradient-Push update -- #

            # Gossip
            ps_result = psga.gossip(gossip_value=ps_n, ps_weight=ps_w)
            ps_n = ps_result['ps_n']
            ps_w = ps_result['ps_w']

            # Update argmin estimate and take a step
            argmin_est = ps_result['avg']
            ps_n = self._gradient_descent_step(ps_n=ps_n,
                                               argmin_est=argmin_est,
                                               itr=itr,
                                               start_time=start_time)

            # -- END Subgradient-Push update -- #

            # Log the varaibles
            if log:
                l_argmin_est.log(argmin_est, itr)
                l_ps_w.log(ps_w, itr)


            # Update the termination flag
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

        pssgd = PushSumSubgradientDescent(objective=objective,
                                          sub_gradient=gradient,
                                          arg_start=x_start,
                                          synch=True,
                                          peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                                          step_size=1e-4,
                                          terminate_by_time=False,
                                          termination_condition=1000,
                                          log=True,
                                          in_degree=2,
                                          num_averaging_itr=1,
                                          constant_step_size=False,
                                          learning_rate=1)

        loggers = pssgd.minimize()
        l_argmin_est = loggers['argmin_est']

        l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=True)

    # Run a demo where nodes minimize a sum of squares function
    demo(num_instances_per_node=5000, num_features=100)
