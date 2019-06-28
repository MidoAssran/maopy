"""
Netowrk Generator script creates graph adjacency matrices.abs
:author: Mido Assran
:description:   Script used to generate strongly-connected bidirectional graphs
                represented by an adjacency matrix. Graphs are then verfied to
                be strongly by performing a BFS from each node in the graph.
                Several graphs are created for each specified graph size.
                A doubly stochastic mixing matrix is constructed from each one
                    Based on the paper (capelliniRandom2009)
                Then the graphs are scored based on the size of corresponding
                mixing matrix's second largest eigenvalue (the largest
                eigenvalue should be '1'). The graph with the best information
                diffusion properties is kept.
"""

import numpy as np
from utils.distributed import score_graph, bfs

CONFIG = {
    'save-fpath': './graphs/',
    'graph-type': 'erdos-renyi',
    'graphs': [
        {'num-nodes': n, 'avg-degree': min(n-1, n // 4),
         'num-trials': 10*n}
        for n in [4, 8, 16, 32, 48, 64, 128]
    ]
}


def main():
    """ The network generator script. """
    save_fpath = CONFIG['save-fpath']
    graph_type = CONFIG['graph-type']
    graph_settings = CONFIG['graphs']

    # Create all graphs in confing file
    for graph in graph_settings:

        # Load graph settings
        num_nodes = graph['num-nodes']
        avg_degree = graph['avg-degree']
        num_trials = graph['num-trials']
        save_fname = save_fpath + graph_type + '_n%s' % num_nodes

        # Define edge probabilities
        edge_probability = avg_degree / (num_nodes - 1)
        edge_probabilties = [1 - edge_probability, edge_probability]
        print('edge-probabilites', edge_probabilties)

        lowest_score = 1.0
        best_graph = None
        # Construct $(num_trials) graphs and choose the best
        for seed in range(num_trials):
            np.random.seed(seed)

            # Create adjacency
            num_unique_edges = int(((num_nodes ** 2) / 2.0)
                                   - (num_nodes / 2.0))
            edges = list(np.random.choice(2, size=num_unique_edges,
                                          p=edge_probabilties))
            a_m = np.identity(num_nodes)
            for i, row in enumerate(a_m):
                for j, _ in enumerate(row[i + 1:]):
                    a_m[i, i + 1 + j] = edges.pop()
                    a_m[i + 1 + j, i] = a_m[i, i + 1 + j]
            adjacency = np.copy(a_m)

            # Check if strongly connected by performing a BFS on graph
            marker = bfs(a_m, num_nodes)

            # Make doubly sochastic - based on paper: (capelliniRandom2009)
            itr = 10
            for _ in range(itr):
                for i, row in enumerate(a_m):
                    a_m[i, :] /= sum(row)
                for j, column in enumerate(np.transpose(a_m)):
                    a_m[:, j] /= sum(column)

            if sum(marker) == num_nodes:
                score = score_graph(a_m)
                if score < lowest_score:
                    lowest_score = score
                    best_graph = adjacency
            else:
                # Graph is not strongly connected => best_graph has not changed
                pass

        print(save_fname, '\n score:', lowest_score, '\n', best_graph)
        np.savez_compressed(save_fname, graph=best_graph)


if __name__ == '__main__':
    main()
