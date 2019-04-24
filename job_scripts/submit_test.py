import os
import sys

command = 'mpirun --mca btl_base_warn_component_unused 0 \
    --report-bindings --bind-to hwthread \
    -n {world_size} python -u main.py \
    --data-file-name qp_data_sg.npz \
    --graph-file-name erdos-renyi_n{world_size}.npz \
    --alg {alg} --lr {lr} --seed 1 --num-steps {steps} \
    --log-dir /async_maopy_playground/qp/n{world_size}/{tag}/'

world_size_list = [2, 4, 8, 16, 32]
world_size_list = [4, 8]
asynch = False
lr = 100.0
tags = {
    'gp': {
        'alg': 'gp',
        'steps': 100,
        'identifiers': ''
    },
    'agp': {
        'alg': 'gp',
        'steps': 100,
        'identifiers': ' --asynch'
    }
}


def main():
    for world_size in world_size_list:
        tag = str(sys.argv[1])
        alg = tags[tag]['alg']
        steps = tags[tag]['steps']
        f_command = command.format(world_size=world_size, alg=alg,
                                   lr=lr, steps=steps, tag=tag)
        f_command += tags[tag]['identifiers']
        os.system(f_command)


if __name__ == '__main__':
    main()
