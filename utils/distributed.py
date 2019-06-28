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

    adjacency_matrix = load_graph(graph_name)
    row, col = adjacency_matrix[rank, :], adjacency_matrix[:, rank]
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
    batch_size = a_m.shape[0]
    objective = lambda x: 0.5 * (np.linalg.norm(a_m.dot(x) - b_v))**2 / batch_size
    gradient = lambda x: a_m.T.dot(a_m.dot(x) - b_v) / batch_size

    return (objective, gradient, arg_start, arg_min)


def load_graph(graph_name):
    """ Load an adjacency graph file. """
    fpath = './graphs/%s' % graph_name
    data = np.load(fpath)
    adjacency_matrix = data['graph']
    return adjacency_matrix


def score_graph(adjacency_matrix):
    """ Compute second largest eig-val. of doubly stochastic mixing matrix. """
    a_m = np.copy(adjacency_matrix)

    # Make doubly sochastic - based on paper: (capelliniRandom2009)
    itr = 10
    for _ in range(itr):
        for i, row in enumerate(a_m):
            a_m[i, :] /= sum(row)
        for j, column in enumerate(np.transpose(a_m)):
            a_m[:, j] /= sum(column)

    arr_lambda = np.linalg.eigvals(a_m)
    for i, l in enumerate(arr_lambda):
        arr_lambda[i] = abs(l)
    arr_lambda = np.sort(arr_lambda)  # sort in increasing order
    return arr_lambda[-2]


def bfs(adjacency, num_nodes):
    """ Breadth first search of a graph. """

    marker = np.zeros(num_nodes)
    marker[0] = 1
    queue = [0]
    while queue:
        node = int(queue.pop())
        for row, val in enumerate(adjacency[:, node]):
            if (marker[row] == 0) and (val == 1):
                marker[row] = 1
                queue.insert(0, row)
    return marker


def main():
    """ Just run stand alone serial utils here. """

    sizes = [2, 4, 8, 16, 32, 64, 128]
    graph_name = 'erdos-renyi_n%s.npz'
    for size in sizes:
        f_graph_name = graph_name % size
        adjacency_matrix = load_graph(f_graph_name)
        lambda_2 = score_graph(adjacency_matrix)
        print(f_graph_name, '\n score:', lambda_2, '\n', adjacency_matrix)


if __name__ == "__main__":
    main()
