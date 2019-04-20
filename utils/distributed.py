""" Utils and general helper functions for running experiments. """

import numpy as np


class Printer:
    def __init__(self, rank, size, comm):
        self.rank = rank
        self.size = size
        self.comm = comm

    def stdout(self, text, synch=False):
        text = '%s: %s' % (self.rank, text)
        if not synch:
            print(text)
            return
        for r in range(self.size):
            self.comm.Barrier()
            if r == self.rank:
                print(text)


def load_peers(graph_name, rank=0, printer=None):
    """ Load a node's peers from an adjacency graph file. """

    fpath = './graphs/%s' % graph_name
    data = np.load(fpath)
    adjacency_matrix = data['graph']
    row, col = adjacency_matrix[rank, :], adjacency_matrix[:, rank]
    del data
    del adjacency_matrix
    peers = [peer for peer, v in enumerate(col) if (peer != rank) and (v == 1)]
    out_degree = len(peers)
    in_degree = int(sum(row)) - 1

    if printer is not None:
        printer.stdout('peers: %s' % peers, synch=True)

    return (peers, in_degree, out_degree)


def load_qp_data(data_name, rank=0, size=1, printer=None):
    """ Load the subset of the data to be used by the local node. """

    fpath = './datasets/%s' % data_name
    data = np.load(fpath)
    argmin_true = data['x_star']
    a_m = data['A']
    b_v = data['b']
    arg_start = data['x_0']
    del data

    # Extract just this node's subset of the data
    split_size = int(a_m.shape[0] // size)
    start_index = int(rank * split_size)
    if rank < (size - 1):
        end_index = int((rank+1) * split_size)
        a_m = a_m[start_index:end_index, :]
        b_v = b_v[start_index:end_index]
    else:
        a_m = a_m[start_index:, :]
        b_v = b_v[start_index:]

    # Print problem size
    if printer is not None:
        printer.stdout('<Local problem size>\n\t %s' % (a_m.shape,), synch=True)

    return (a_m, b_v, arg_start, argmin_true)


def load_least_squares(data_name, rank=0, size=1, printer=None):

    a_m, b_v, arg_start, arg_min = load_qp_data(data_name, rank, size, printer)
    objective = lambda x: 0.5 * (np.linalg.norm(a_m.dot(x) - b_v))**2
    gradient = lambda x: a_m.T.dot(a_m.dot(x) - b_v)

    return (objective, gradient, arg_start, arg_min)
