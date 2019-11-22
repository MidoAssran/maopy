"""
Pull Gossip Averager class for parallel averaging using row stochastic mixing

:author: Mido Assran
:description: Distributed pull-based gossip using row stochastic mixing.
              Based on the paper (todo:fill-in)
"""

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

    def __init__(self, peers=None, in_degree=SIZE):
        """ Initialize the distributed averaging settings """

        # Break on all numpy warnings
        np.seterr(all='raise')

        # Set peers to all if not told who peers are
        if not peers:
            peers = [i for i in range(SIZE) if i != UID]
        self.peers = peers

        self.out_degree = len(self.peers)
        self.in_degree = in_degree
        self.info_list = []
        self.out_reqs = []

    def push_messages_to_peers(self, peers, ps_n):
        """
        Send scaled push sum numerator to peers.

        :type peers: list[int]
        :type consensus_column: list[float]
        :type ps_n: float
        :rtype: void
        """
        for i, peer_uid in enumerate(peers):
            req = COMM.Ibsend(ps_n, dest=peer_uid)
            self.out_reqs.append(req)

    def receive_asynchronously(self, gossip_value):
        """
        Probe buffer (non-blocking) & and retrieve all messages until the receive buffer is empty.

        :rtype: np.array[float] or float
        """
        rcvd_data = [gossip_value]
        info = [MPI.Status()]
        while COMM.Iprobe(source=MPI.ANY_SOURCE, status=info[0]):
            self.info_list.append(info[0])
            info[0] = MPI.Status()
            data = np.empty(gossip_value.shape, dtype=np.float64)
            COMM.Recv(data, info[0].source)
            rcvd_data.append(data)

        return rcvd_data

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

        # Extra logic to determine whether all out-comms are done
        done_list = []
        done_sending = True
        for i, req in enumerate(self.out_reqs):
            if req.test()[0]:
                done_list.append(i)
            else:
                done_sending = False
                break
        if done_sending:
            self.out_reqs.clear()

        return gossip_value


if __name__ == "__main__":

    def demo(gossip_value):
        """
        Demo of the use of the PullGossipAverager class

        To run the demo, run the following form the command line:
            mpiexec -n $(num_nodes) python -m pull_gossip_averaging
        """

        # Initialize averager
        plga = PullGossipAverager(peers=[(UID + 1) % SIZE, (UID + 2) % SIZE],
                                  in_degree=2)

        for itr in range(10):
            gossip_value = plga.gossip(gossip_value)
            # report gossip-value
            print('%s: %s' % (UID, gossip_value))

    # Run a demo where nodes average their unique IDs
    demo(gossip_value=UID)
