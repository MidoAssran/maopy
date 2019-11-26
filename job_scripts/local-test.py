import os

command = 'mpirun --mca btl_base_warn_component_unused 0 \
    --report-bindings --bind-to hwthread \
    -n {world_size} python -u main.py \
    --data-file-name covtype.data --experiment softmax \
    --graph-file-name {graph}_n{world_size}.npz \
    --alg {alg} --lr {lr} --seed 1 --num-steps {steps} --tau {tau} \
    --log-dir /tmp/softmax/n{world_size}/step-size{lr}/{graph}/{tag}/'

# --data-file-name qp_data_sg.npz \

tau = 32
# size_list = [4, 8, 16, 32, 64, 128]
size_list = [4]
lr_list = [1e0]
graph_list = ['erdos-renyi']
tags = {
    # 'gp': {
    #     'alg': 'gp',
    #     'steps': lambda s: 10000,
    #     'identifiers': ''
    # },
    'agp': {
        'alg': 'gp',
        'steps': lambda s: 600,
        'identifiers': ' --asynch'
    },
    # 'ep': {
    #     'alg': 'ep',
    #     'steps': lambda s: 10000,
    #     'identifiers': ''
    # },
    # 'pd': {
    #     'alg': 'pd',
    #     'steps': lambda s: 10000,
    #     'identifiers': ''
    # }
    # 'asy-sonata': {
    #     'alg': 'asy-sonata',
    #     'steps': lambda s: 10,
    #     'identifiers': ''
    # }
}


def main():
    for size in size_list:
        for tag in tags:
            for graph in graph_list:
                alg = tags[tag]['alg']
                steps = tags[tag]['steps'](tau)
                for lr in lr_list:
                    f_command = command.format(world_size=size, alg=alg, lr=lr,
                                               steps=steps, tag=tag, tau=tau,
                                               graph=graph)
                    f_command += tags[tag]['identifiers']
                    try:
                        os.system(f_command)
                    except Exception as e:
                        print(e)


if __name__ == '__main__':
    main()
