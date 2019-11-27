"""
Gradient-Push for distributed optimization using column-stochastic mixing

:author: Mido Assran
:description: Distributed optimization using column-stochastic mixing.
              Based on the paper (todo)
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


class GradientPush(object):
    """
    Distributed optimization using column-stochastic mixing.
    """

    def __init__(self,
                 objective,
                 sub_gradient,
                 arg_start,
                 asynch=False,
                 peers=[(UID+1) % SIZE],
                 step_size=1e-2,
                 max_itr=100,
                 max_time_sec=100,
                 in_degree=SIZE,
                 tau_proc=40,
                 tau_msg=100,
                 log=True):
        """ Initialize the gossip optimization settings. """

        self.objective = objective
        self.sub_gradient = sub_gradient
        self.argmin_est = np.array(arg_start)

        # If asynchronous, tau_proc and tau_msg are used to monitor processing
        # and message delays, and block the agent if these delay thresholds are
        # exceeded
        self.asynch = asynch
        self.tau_proc = tau_proc
        self.tau_msg = tau_msg - (tau_proc - 1)

        self.peers = peers
        self.step_size = step_size

        self.max_itr = max_itr
        self.max_time = max_time_sec

        # Termination function
        def terminator(itr=0, time_sec=0):
            return (itr > self.max_itr) or (time_sec > self.max_time)
        self.the_terminator = terminator

        self.in_degree = in_degree
        self.log = log

        self.psga = PushSumAverager(peers, in_degree)

    def __setattr__(self, name, value):
        super(GradientPush, self).__setattr__(name, value)

        if (name == 'peers') or (name == 'in_degree'):
            # Updating gossip averager property with $(name)
            try:
                psga = super(GradientPush, self).__getattribute__('psga')
                setattr(psga, name, value)
            except AttributeError:
                # Do nothing if the instance hasn't been created yet
                pass

    def minimize(self):
        """
        Minimize the objective using GradientPush

        Procedure:
        1) ps_n = ps_n - step_size * sub_gradient(argmin_est)
        1) Gossip: push_sum_gossip([ps_n, ps_w])
        2) Update: ps_result = push_sum_gossip([ps_n, ps_w])
            2.a) ps_n = ps_result[ps_n]
            2.b) ps_w = ps_result[ps_w]
            2.c) argmin_est = ps_n / ps_w
        3) Repeat until termination_condition specific by "the_terminator"

        :rtype:
            log is True: dict("argmin_est": GossipLogger,
                              "ps_w": GossipLogger)

            log is False: dict("argmin_est": float,
                               "objective": float,
                               "sub_gradient": float)
        """

        # ----Initialize gradient-push---- #
        ps_n = self.argmin_est
        ps_w = 1.0
        argmin_est = ps_n / ps_w

        itr = 0
        start_time = time.time()
        # -- used for cleaning up MPI
        num_rcvd, num_sent = 0, 0
        # -- keep track of message staleness
        staleness = 0
        # -- keep track of relative processing delays
        tau = 0
        barrier_req = COMM.Ibarrier()

        if self.log:
            # -- log the argmin estimate
            l_argmin_est = GossipLog()
            l_argmin_est.log(argmin_est, itr)
            # -- log the push sum weight
            l_ps_w = GossipLog()
            l_ps_w.log(ps_w, itr)

        # -- start optimization at the same time
        COMM.Barrier()
        print('%s: starting optimization...' % UID)
        np.random.seed(UID)

        # Optimization loop
        while not self.the_terminator(itr, time.time()-start_time):

            if not self.asynch:
                COMM.Barrier()

            # Update iteration
            itr += 1

            # -- local descent step
            ps_n -= self.step_size * self.sub_gradient(argmin_est)

            # -- push-sum gossip
            just_probe = ps_w < 1e-5  # -- prevent numerical instability
            ps_result = self.psga.gossip(ps_n, ps_w, just_probe, self.asynch)
            ps_n = ps_result['ps_n']
            ps_w = ps_result['ps_w']
            num_rcvd += ps_result['num_rcvd']
            if not just_probe:
                num_sent += len(self.peers)

            # -- keep track of (and bound) message staleness
            staleness = 0 if ps_result['num_rcvd'] > 0 else staleness + 1
            if staleness > self.tau_msg:
                print('%s: messages too stale... waiting' % UID)
                ps_result = self.psga.gossip(ps_n, ps_w,
                                             just_probe=True,
                                             asynch=False)
            argmin_est = ps_result['avg']
            ps_n = ps_result['ps_n']
            ps_w = ps_result['ps_w']
            num_rcvd += ps_result['num_rcvd']

            # -- record-keeping to clean up MPI at the end
            print('%s: jp(%s) %s' % (UID, just_probe, ps_w))
            print('%s: rcvd/sent(%s/%s) %s' % (UID, num_rcvd, num_sent, ps_w))

            # -- keep track of (and bound) relative processing delays
            if barrier_req.test()[0]:
                tau = 1
                barrier_req = COMM.Ibarrier()
            else:
                tau += 1
                if tau >= self.tau_proc:
                    tau = 1
                    print('%s: peer is too far behind... waiting' % UID)
                    timeout = time.time()
                    while not barrier_req.test()[0]:
                        barrier_time = time.time() - timeout
                        if barrier_time > 30.:
                            print('%s: proc. barrier timeout... quiting' % UID)
                            break

            # -- log updated variables
            if self.log:
                l_argmin_est.log(argmin_est, itr)
                l_ps_w.log(ps_w, itr)

        COMM.Barrier()
        if self.asynch:
            # Fetch any lingering message
            print('%s: Fetching lingering msgs...' % (UID))
            timeout = time.time()
            ps_result = self.psga.gossip(ps_n, ps_w,
                                         just_probe=True,
                                         asynch=False)
            argmin_est = ps_result['avg']
            ps_n = ps_result['ps_n']
            ps_w = ps_result['ps_w']
            num_rcvd += ps_result['num_rcvd']
            print('%s: rcvd/sent(%s/%s) %s' % (UID, num_rcvd, num_sent, ps_w))
            barrier_time = time.time() - timeout
            # -- log updated variables
            if self.log:
                itr += 1
                l_argmin_est.log(argmin_est, itr,
                                 time_offset=(-barrier_time),
                                 force=True)

        self.argmin_est = argmin_est

        if self.log:
            return {'argmin_est': l_argmin_est,
                    'ps_w': l_ps_w}
        else:
            return {'argmin_est': argmin_est,
                    'objective': self.objective(argmin_est),
                    'sub_gradient': self.sub_gradient(argmin_est)}


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

        gradpush = GradientPush(objective=objective,
                                sub_gradient=gradient,
                                arg_start=x_start,
                                peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                                step_size=1e-5,
                                max_itr=1e100,
                                max_time_sec=5.0,
                                in_degree=2,
                                asynch=True,
                                tau_proc=32)

        loggers = gradpush.minimize()
        l_argmin_est = loggers['argmin_est']

        print('%s: (start: %s)(finish: %s)'
              % (UID, obj_start, globjective(l_argmin_est.gossip_value)))

    # Run a demo where nodes minimize a sum of squares function
    demo(num_instances_per_node=5000, num_features=100)
