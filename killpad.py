import os

if __name__ == '__main__':
    # command = 'sbatch {fpath}'
    command = 'scancel -n {fpath}'
    fpath = '{tag}-n{num_nodes}-{graph}-lr{lr}'
    world_size_list = [40]
    tag_list = [
        'ep', 'gp', 'pd'
    ]
    graph_list = ['erdos-renyi', 'ring']
    lr_list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    for size in world_size_list:
        for tag in tag_list:
            for gph in graph_list:
                for lr in lr_list:
                    f_fpath = fpath.format(tag=tag, num_nodes=size, graph=gph,
                                           lr=lr)
                    f_command = command.format(fpath=f_fpath)
                    print(f_command)
                    os.system(f_command)
