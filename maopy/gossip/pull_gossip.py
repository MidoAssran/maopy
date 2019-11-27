"""
Pull Gossip Averager class for parallel averaging using row stochastic mixing

:author: Mido Assran
:description: Distributed pull-based gossip using row stochastic mixing.
              Based on the paper (todo:fill-in)
"""

import time
from collections import defaultdict

from mpi4py import MPI
import numpy as np

from .gossip_comm import GossipComm

# Message passing and network variables
COMM = GossipComm.comm
SIZE = GossipComm.size
UID = GossipComm.uid
NAME = GossipComm.name


class PullGossipAverager(object):
    """
    Distributed row stochastic averaging

    :param peers: UniqueIDs of neighbouring peers in network (used for comm.)
    :param in_degree: Num. messages to expect in each itr.
    """

    def __init__(self, peers=[(UID+1) % SIZE]):
        """ Initialize the distributed averaging settings """

        # # Break on all numpy warnings
        # np.seterr(all='raise')

        self.peers = peers

        self.out_degree = len(self.peers)
        self.info_list = []
        self.out_reqs = defaultdict(list)

    def push_messages_to_peers(self, peers, ps_n):
        """
        Send scaled push sum numerator to peers.

        :type peers: list[int]
        :type consensus_column: list[float]
        :type ps_n: float
        :rtype: void
        """
        for i, peer_uid in enumerate(peers):
            # -- check if last message to peer was sent
            done_indices = []
            for i, req in enumerate(self.out_reqs[peer_uid]):
                if not req.test()[0]:
                    continue
                done_indices.append(i)
            for index in sorted(done_indices, reverse=True):
                    del self.out_reqs[peer_uid][index]
            req = COMM.Ibsend(ps_n, dest=peer_uid, tag=1573)
            self.out_reqs[peer_uid].append(req)

    def receive_asynchronously(self, gossip_value):
        """
        Probe buffer (non-blocking) & and retrieve all messages until the
        receive buffer is empty.

        :rtype: np.array[float] or float
        """
        rcvd_data = defaultdict(list)
        rcvd_data[UID].append(gossip_value)

        info = MPI.Status()
        while COMM.Iprobe(source=MPI.ANY_SOURCE, status=info, tag=1573):
            self.info_list.append(info)
            data = np.empty(gossip_value.shape, dtype=np.float64)
            COMM.Recv(data, info.source, tag=1573)
            rcvd_data[info.source].append(data)
            info = MPI.Status()

        new_data = []
        for peer in rcvd_data:
            msg_list = np.array(rcvd_data[peer])
            # assume messages arrive in-order (take most recent)
            msg = msg_list[-1]
            print('%s: from %s (num-msgs %s)'
                  % (UID, peer, len(msg_list)))
            new_data.append(msg)

        return new_data

    def gossip(self, gossip_value, just_probe=False):
        """
        Perform the distributed gossip averaging

        :type gossip_value: float or np.array[float]
        :rtype: float or np.array[float]
        """

        gossip_value = np.array(gossip_value, dtype=np.float64)

        if not just_probe:
            self.push_messages_to_peers(self.peers, gossip_value)
        rcvd_data = self.receive_asynchronously(gossip_value)
        gossip_value = sum(rcvd_data)/len(rcvd_data)

        return gossip_value


if __name__ == "__main__":

    def demo(gossip_value):
        """
        Demo of the use of the PullGossipAverager class

        To run the demo, run the following form the command line:
            mpiexec -n $(num_nodes) python -m maopy.pull_gossip
        """

        # Initialize averager
        plga = PullGossipAverager(peers=[(UID + 1) % SIZE, (UID + 2) % SIZE])

        for itr in range(100):
            gossip_value = plga.gossip(gossip_value)
            # report gossip-value
            print('%s: %s' % (UID, gossip_value))

        # Fetch any lingering message
        COMM.Barrier()
        print('%s: Fetching lingering msgs...' % (UID))
        timeout = time.time()
        while (time.time() - timeout) <= 10.:
            gossip_value = plga.gossip(gossip_value,
                                       just_probe=True)
            time.sleep(0.5)

        # make sure true mean has not changed (aggregate reulsts)
        print('%s: %s' % (UID, gossip_value))
        COMM.Barrier()
        print('%s: sanity check' % UID)
        total = np.empty(gossip_value.shape, dtype=np.float64)
        COMM.Allreduce(gossip_value, total, op=MPI.SUM)
        print('%s: network-wide average %s' % (UID, total / SIZE))

    # Run a demo where nodes average their unique IDs
    demo(gossip_value=UID)
