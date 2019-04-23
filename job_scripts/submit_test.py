import os

command = 'mpirun --mca btl_base_warn_component_unused 0 \
    --report-bindings --bind-to hwthread \
    -n {world_size} python -u main.py \
    --data-file-name qp_data_sg.npz \
    --graph-file-name erdos-renyi_n{world_size}.npz \
    --alg {alg} --lr {lr} --seed 1 --num-steps {steps} \
    --log-dir /async_maopy_playground/qp/n{world_size}/{tag}/'

world_size_list = [2, 4, 8, 16, 32]
asynch = True
# steps = 100
steps = 10
lr = 0.0001
alg = 'gp'


def main():
    for world_size in world_size_list:
        tag = 'a' + alg if asynch else alg
        f_command = command.format(world_size=world_size, alg=alg,
                                   lr=lr * world_size,
                                   steps=steps, tag=tag)
        if asynch:
            f_command += ' --asynch'
        os.system(f_command)


if __name__ == '__main__':
    main()
