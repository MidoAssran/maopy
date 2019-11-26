"""
Labor-Bureau generates jobs (scripts)
"""

srun_template = '''#!/bin/bash
#SBATCH --job-name={job_tag}
#SBATCH --output=/checkpoint/%u/async_maopy_playground/{job_tag}.out
#SBATCH --error=/checkpoint/%u/async_maopy_playground/{job_tag}.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks={size}
#SBATCH --mem-per-cpu=5000
#SBATCH --gres=gpu:0
#SBATCH --time={rtime}
#SBATCH --partition=learnfair

module purge
module load anaconda3
source deactivate
source activate /private/home/massran/.conda/envs/agp-env

mpirun --mca btl_base_warn_component_unused 0 \\
    --mca btl_openib_warn_default_gid_prefix 0 \\
    -n {size} python -u main.py  --tau {tau} \\
    --data-file-name 'qp_data_sg.npz' \\
    --graph-file-name '{graph}_n{size}.npz' \\
    --alg {alg} --lr {lr} --seed 1 --num-steps {steps} \\
    --log-dir '/async_maopy_playground/slurm/nll/n{size}/step-size{lr}/{graph}/{tag}/' '''

# Sys.-Run Config
rtime = '00:40:00'
tau = 4
world_size_list = [2, 4, 8, 16, 32, 64, 128]
lr_list = [1e0]
graph_list = ['erdos-renyi']
alg_list = {
    'asy-sonata': {
        'alg': 'asy-sonata',
        'steps': 1200,
        'identifiers': ''
    },
    'gp': {
        'alg': 'gp',
        'steps': 35000,
        'identifiers': ''
    },
    'agp': {
        'alg': 'gp',
        'steps': 1200,
        'identifiers': ' --asynch'
    },
    'pd': {
        'alg': 'pd',
        'steps': 35000,
        'identifiers': ''
    },
    'ep': {
        'alg': 'ep',
        'steps': 35000,
        'identifiers': ''
    }
}


def create_jobs():
    tag_template = '{tag}-n{size}-{gph}-lr{lr}'
    for size in world_size_list:
        for tag in alg_list:
            alg = alg_list[tag]['alg']
            steps = alg_list[tag]['steps']
            for graph in graph_list:
                for lr in lr_list:
                    f_tag = tag_template.format(tag=tag, size=size,
                                                gph=graph, lr=lr)
                    print(f_tag)
                    job_script = srun_template.format(
                        job_tag=f_tag, size=size, rtime=rtime,
                        alg=alg, lr=lr, steps=steps, tag=tag,
                        graph=graph, tau=tau)
                    job_script += alg_list[tag]['identifiers']
                    fname = 'submit_' + f_tag + '.sh'
                    with open(fname, 'w') as f:
                        f.write(job_script)


if __name__ == '__main__':
    """ Welcome to the labor bureau... we're going to create jobs (scripts) """
    create_jobs()
    print('Done.')
