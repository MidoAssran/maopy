"""
Push Sum Gossip DIGing class for parallel optiization using column stochastic mixing.

:author: Mido Assran
:description: Distributed optimization using column stochastic mixing and gradient tracking.
              Based on the paper (nedich2016Achieving)
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


class PushDIGing(PushSumOptimizer):
    """ Distributed optimization with gradient tracking and column stochastic mixing. """

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

        super(PushDIGing, self).__init__(objective=objective,
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
        1) Update: ps_argmin_n -= step_size * ps_grad_n
        2) Gossip: push_sum_gossip([ps_argmin_n, ps_grad_n, ps_w])
        3) Update: ps_result = push_sum_gossip([ps_argmin_n, ps_grad_n, ps_w])
            3.a) Update: ps_argmin_n = ps_result[ps_argmin_n]
            3.b) Update: ps_grad_n = ps_result[ps_grad_n]
            3.c) Update: ps_w = ps_result[ps_w]
            3.d) Update: argming_est = ps_argmin_n / ps_w
            3.e) Update: ps_grad_n += sub_gradient(argmin_est@(k)) - sub_gradient(argmin_est@(k-1))
        3) Repeat until completed $(termination_condition) itr. or time (depending on settings)

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """
        super(PushDIGing, self).minimize()


        psga = self.ps_averager

        step_size = self.step_size
        gradient = self.sub_gradient
        objective = self.objective

        #----Initialize push diging----#
        argmin_est = self.argmin_est
        ps_argmin_n = self.argmin_est
        ps_w = 1.0
        ps_grad_n_k = gradient(argmin_est)
        grad_km1 = ps_grad_n_k

        # Setup loop parameters
        itr = 0
        log = self.log
        if log:
            from .gossip_log import GossipLog
            l_argmin_est = GossipLog() # Log the argmin estimate
            l_ps_w = GossipLog() # Log the push sum weight
            l_argmin_est.log(argmin_est, itr)
            l_ps_w.log(ps_w, itr)

        # Setup the termination condition (time or iteration)
        if self.terminate_by_time is False:
            num_gossip_itr = self.termination_condition
            condition = itr < num_gossip_itr
        else:
            end_time = time.time() + self.termination_condition
            condition = time.time() < end_time

        # Start optimization at the same time
        COMM.Barrier()

        # Optimization loop
        while condition:

            # if UID == 0:
            #     time.sleep(.1)

            # Update iteration
            itr += 1

            # -- START PushDIGing update -- #

            # Take a step and gossip
            ps_argmin_n -= step_size * ps_grad_n_k
            gossip_vector = np.append(ps_argmin_n, ps_grad_n_k)
            ps_result = psga.gossip(gossip_value=gossip_vector, ps_weight=ps_w)
            print('%s: itr(%s) %s' % (UID, itr, ps_w))

            try:
                # Update argmin estimate
                ps_w = ps_result['ps_w']
                ps_n = np.array(ps_result['ps_n'])
                ps_argmin_n = ps_n[:ps_argmin_n.size].reshape(ps_argmin_n.shape)
                argmin_est = ps_argmin_n / ps_w

                # Update gradient tracking estimate
                grad_k = gradient(argmin_est)
                ps_grad_n_k = ps_n[ps_argmin_n.size:].reshape(ps_grad_n_k.shape)
                ps_grad_n_k += grad_k - grad_km1
                # Update the past gradient (at itr. k minus 1)
                grad_km1 = grad_k
            except Exception as e:
                print('%s: itr(%s) error: %s' % (UID, itr, e))

            # -- END PushDIGing update -- #

            # Log the varaibles
            if log:
                l_argmin_est.log(argmin_est, itr)
                l_ps_w.log(ps_w, itr)


            # Update the termination flag
            if self.terminate_by_time is False:
                condition = itr < num_gossip_itr
            else:
                condition = time.time() < end_time

            if self.synch is True:
                COMM.Barrier()


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

        pd = PushDIGing(objective=objective,
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

        loggers = pd.minimize()
        l_argmin_est = loggers['argmin_est']

        l_argmin_est.print_gossip_value(UID, label='argmin_est', l2=True)

    # Run a demo where nodes minimize a sum of squares function
    demo(num_instances_per_node=5000, num_features=100)
