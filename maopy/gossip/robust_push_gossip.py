"""
Robust Push Gossip Averaging class for parallel averaging using column
stochastic mixing

:author: Mido Assran
:description: Distributed push-based gossip using column stochastic mixing.
              Based on the paper (todo:fill-in)
"""

from mpi4py import MPI
import numpy as np
from collections import defaultdict

from .gossip_comm import GossipComm

# Message passing and network variables
COMM = GossipComm.comm
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name


class RobustPushAverager(object):
    """
    Distributed column stochastic averaging, robust to dropped messages.
    However, does not keep track of push-sum weight.

    :param peers: UniqueIDs of neighbouring peers in network (used for comm.)
    """

    def __init__(self, peers=None):
        """ Initialize the distributed averaging settings """

        # # Break on all numpy warnings
        # np.seterr(all='raise')

        # Set peers to all if not told who peers are
        if not peers:
            peers = [i for i in range(SIZE) if i != UID]
        self.peers = peers

        self.out_degree = len(self.peers)
        self.info_list = []
        self.out_buffer = {}
        self.in_buffer = {}
        self.out_reqs = defaultdict(list)

    def make_stochastic_weight_column(self):
        """ Creates a column of weights for the mixing matrix. """
        column = {}
        lo_p = 1.0 / (self.out_degree + 1.0)
        out_p = [1.0 / (self.out_degree + 1.0) for _ in range(self.out_degree)]
        column['lo_p'] = lo_p
        column['out_p'] = out_p
        return column

    def push_messages_to_peers(self, peers, consensus_column, ps_n):
        """
        Send scaled push sum numerator to peers.

        :type peers: list[int]
        :type consensus_column: list[float]
        :type ps_n: float
        :rtype: void
        """
        for cc_w, peer_uid in zip(consensus_column, peers):

            # -- check if last message to peer was sent
            done_indices = []
            for i, req in enumerate(self.out_reqs[peer_uid]):
                if not req.test()[0]:
                    continue
                done_indices.append(i)
            for index in sorted(done_indices, reverse=True):
                    del self.out_reqs[peer_uid][index]

            # -- send message to peer
            push_message = cc_w * ps_n
            if peer_uid not in self.out_buffer:
                self.out_buffer[peer_uid] = push_message
            else:
                self.out_buffer[peer_uid] += push_message
            req = COMM.Ibsend(self.out_buffer[peer_uid], dest=peer_uid,
                              tag=7462)
            self.out_reqs[peer_uid].append(req)

    def receive_asynchronously(self, gossip_value):
        """
        Probe buffer (non-blocking) & and retrieve all messages until the
        receive buffer is empty.

        :rtype: np.array[float] or float
        """
        rcvd_data = defaultdict(list)

        info = MPI.Status()
        while COMM.Iprobe(source=MPI.ANY_SOURCE, status=info, tag=7462):
            self.info_list.append(info)
            data = np.empty(gossip_value.shape, dtype=np.float64)
            COMM.Recv(data, info.source, tag=7462)
            if info.source not in self.in_buffer:
                self.in_buffer[info.source] = np.zeros(gossip_value.shape,
                                                       dtype=np.float64)
            rcvd_data[info.source].append(data)
            info = MPI.Status()

        new_data = np.zeros(gossip_value.shape, dtype=np.float64)
        for peer in rcvd_data:
            msg_list = np.array(rcvd_data[peer])
            # -- assume messages arrive in-order (take most recent)
            msg = msg_list[-1]
            # -- new data is diff between new msg and whats in our buffer
            new_data += msg - self.in_buffer[peer]
            # -- update buffer with the new msg
            self.in_buffer[peer] = msg

        return new_data

    def gossip(self, gossip_value, just_probe=False):
        """
        Perform the distributed gossip averaging

        :type gossip_value: float or np.array[float]
        :rtype: float or np.array[float]
        """

        gossip_value = np.array(gossip_value, dtype=np.float64)

        column = self.make_stochastic_weight_column()
        out_p = column['out_p']  # vector
        lo_p = column['lo_p']  # scalar

        if not just_probe:
            self.push_messages_to_peers(self.peers, out_p, gossip_value)
            gossip_value *= lo_p
        gossip_value += self.receive_asynchronously(gossip_value)

        return gossip_value


if __name__ == "__main__":
    import time

    def demo(gossip_value):
        """
        Demo of the use of the PullGossipAverager class

        To run the demo, run the following form the command line:
            mpiexec -n $(num_nodes) python -m maopy.robust_push_gossip
        """

        # Initialize averager
        rpga = RobustPushAverager(peers=[(UID + 1) % SIZE, (UID + 2) % SIZE])

        gossip_value = np.append(gossip_value, 1.)

        for itr in range(1000):
            gossip_value = rpga.gossip(gossip_value)
            # report gossip-value
            print('%s: %s' % (UID, gossip_value[:-1].item()/gossip_value[-1]))

        # Fetch any lingering message
        COMM.Barrier()
        print('%s: Fetching lingering msgs...' % (UID))
        timeout = time.time()
        while (time.time() - timeout) <= 10.:
            gossip_value = rpga.gossip(gossip_value,
                                       just_probe=True)
            time.sleep(0.5)

        # make sure true mean has not changed (aggregate reulsts)
        COMM.Barrier()
        print('%s: sanity check' % UID)
        total = np.empty(gossip_value.shape, dtype=np.float64)
        COMM.Allreduce(gossip_value, total, op=MPI.SUM)
        print('%s: network-wide average %s'
              % (UID, total[:-1].item()/(total[-1])))

    # Run a demo where nodes average their unique IDs
    demo(gossip_value=UID)
