import os

if __name__ == '__main__':
    command = 'sbatch {fpath}'
    fpath = 'job_scripts/submit_{tag}-nodes{num_nodes}.sh'
    tag_list = [
        'agp', 'gp', 'pd', 'ep'
    ]
    world_size_list = [2, 4, 8, 16, 32, 64, 128]
    for size in world_size_list:
        for tag in tag_list:
            f_fpath = fpath.format(tag=tag, num_nodes=size)
            f_command = command.format(fpath=f_fpath)
            os.system(f_command)
