"""
Labor-Bureau generates jobs (scripts)
"""

srun_template = '''#!/bin/bash
#SBATCH --job-name={job_tag}
#SBATCH --output=/checkpoint/%u/async_maopy_playground/{job_tag}.out
#SBATCH --error=/checkpoint/%u/async_maopy_playground/{job_tag}.err
#SBATCH --nodes={size}
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time={rtime}
#SBATCH --partition=dev

module purge
module load anaconda3
module load openmpi
source deactivate
source activate /private/home/massran/.conda/envs/agp

mpirun --report-bindings --oversubscribe --bind-to hwthread \\
    --mca btl_base_warn_component_unused 0 \\
    --mca btl_openib_warn_default_gid_prefix 0 \\
    python -u main.py  \\
    --data-file-name 'qp_data_sg.npz' \\
    --graph-file-name 'erdos-renyi_n{size}.npz' \\
    --alg {alg} --lr {lr} --seed 1 --num-steps {steps} \\
    --log-dir '/async_maopy_playground/qp/n{size}/{tag}/' '''

# Sys.-Run Config
world_size_list = [2, 4, 8, 16, 32, 64, 128]
ref_lr = 0.0001
alg_list = {
    'gp': {
        'alg': 'gp',
        'steps': 100,
        'asynch': False
    },
    'agp': {
        'alg': 'gp',
        'steps': 10,
        'asynch': True
    },
    'pd': {
        'alg': 'pd',
        'steps': 100,
        'asynch': False
    },
    'ep': {
        'alg': 'ep',
        'steps': 100,
        'asynch': False
    }

}


def create_jobs():
    tag_template = '{tag}-nodes{size}'
    for tag in alg_list:
        asynch = alg_list[tag]['asynch']
        steps = alg_list[tag]['steps']
        alg = alg_list[tag]['alg']
        for size in world_size_list:
            lr = ref_lr * size
            f_tag = tag_template.format(tag=tag, size=size)
            print(f_tag)
            job_script = srun_template.format(
                job_tag=f_tag, size=size, rtime='00:20:00',
                alg=alg, lr=lr, steps=steps, tag=tag)
            if asynch:
                job_script += ' --asynch'
            fname = 'submit_' + f_tag + '.sh'
            with open(fname, 'w') as f:
                f.write(job_script)


if __name__ == '__main__':
    """ Welcome to the labor bureau... we're going to create jobs (scripts) """
    create_jobs()
    print('Done.')
