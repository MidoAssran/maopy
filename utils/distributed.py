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
        printer.stdout('<Local problem size>\n\t %s' % (a_m.shape,),
                       synch=True)

    return (a_m, b_v, arg_start, argmin_true)


def load_least_squares(data_name, rank=0, size=1, printer=None):

    a_m, b_v, arg_start, arg_min = load_qp_data(data_name, rank, size, printer)
    batch_size = a_m.shape[0]
    objective = lambda x: 0.5 * (np.linalg.norm(a_m.dot(x) - b_v))**2 / batch_size
    gradient = lambda x: a_m.T.dot(a_m.dot(x) - b_v) / batch_size

    return (objective, gradient, arg_start, arg_min)


def load_cotype_data(data_name, rank=0, size=1, printer=None):
    """ Load the subset of the data to be used by the local node. """

    fpath = './datasets2/%s' % data_name
    raw_data = np.genfromtxt(fpath, delimiter=',')
    feature_matrix, target_vector = raw_data[:, :-1], np.array(raw_data[:, -1],
                                                               dtype=np.int32)
    target_vector -= 1  # reindex class-labels starting from 0
    num_classes = 7
    target_matrix = np.zeros([feature_matrix.shape[0], num_classes])
    target_matrix[np.arange(feature_matrix.shape[0]), target_vector] = 1.0

    # Hacky fix- get rid of last 20 samples to make sure all networks
    # (2, 4, 8, 16, 32, 64, 128) solve the same objective, otherwise might
    # need to change push-sum weight initializations or re-normalize objs
    feature_matrix = feature_matrix[:-20, :]
    target_matrix = target_matrix[:-20]

    # Standardize the non-binary features
    for i in range(10):
        feature_matrix[:, i] -= np.average(feature_matrix[:, i])
        feature_matrix[:, i] /= np.std(feature_matrix[:, i])

    split_size = feature_matrix.shape[0] // size
    start_index = int(rank * split_size)
    if rank < (size - 1):
        end_index = int((rank + 1) * split_size)
        feature_matrix = feature_matrix[start_index:end_index, :]
        target_matrix = target_matrix[start_index:end_index]
    else:
        feature_matrix = feature_matrix[start_index:, :]
        target_matrix = target_matrix[start_index:]

    if printer is not None:
        printer.stdout('<Local problem size>\n\t %s' % feature_matrix.shape[0],
                       synch=True)

    return feature_matrix, target_matrix


def load_softmax(data_name, rank=0, size=1, printer=None):
    """ Return the local node's softmax function for a given dataset. """

    feature_matrix, target_matrix = load_cotype_data(data_name, rank, size,
                                                     printer)
    arg_start = np.random.randn(target_matrix.shape[1],
                                feature_matrix.shape[1])

    # Regularization parameter
    reg_param = 1e-4

    def objective(weight_matrix):
        """ Compute the negative log likelihood of the softmax function. """

        linear_product = weight_matrix.dot(np.transpose(feature_matrix))
        max_linear_product = np.max(linear_product)
        alpha_matrix = np.exp(linear_product - max_linear_product)
        normalizer_vector = np.sum(alpha_matrix, axis=0)
        softmax_matrix = np.divide(alpha_matrix, normalizer_vector)

        # This is like diag(target_matrix.dot(np.log(softmax_matrix)))
        # but more efficient
        log_likelihood_vector = np.einsum('ij,ji->i', target_matrix,
                                          np.log(softmax_matrix))

        # Construct a regularizer
        regularizer = (reg_param / 2.0) * np.linalg.norm(weight_matrix,
                                                         ord='fro')**2

        # Return negative log-likelihood
        objective_scalar = regularizer - (np.sum(log_likelihood_vector)
                                          / feature_matrix.shape[0])
        return objective_scalar

    def gradient(weight_matrix):
        """ Compute the gradient of the NLL of softmax """

        linear_product = weight_matrix.dot(np.transpose(feature_matrix))
        max_linear_product = np.max(linear_product)
        alpha_matrix = np.exp(linear_product - max_linear_product)
        normalizer_vector = np.sum(alpha_matrix, axis=0)
        softmax_matrix = np.divide(alpha_matrix, normalizer_vector)

        num_classes = target_matrix.shape[1]
        gradient_vector = []
        for k in range(num_classes):
            product_vector = softmax_matrix[k, :] - target_matrix[:, k]
            k_gradient = np.transpose(product_vector).dot(feature_matrix)
            gradient_vector.append(k_gradient)

        # Construct gradient of regularization term
        regularizer = reg_param * sum(sum(weight_matrix))
        # Return gradient
        gradient_vector = regularizer + (np.array(gradient_vector)
                                         / feature_matrix.shape[0])
        return gradient_vector

    return (objective, gradient, arg_start)


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
