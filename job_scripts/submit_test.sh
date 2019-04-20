#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/checkpoint/%u/async_maopy_playground/test.out
#SBATCH --error=/checkpoint/%u/async_maopy_playground/test.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --partition=scavenge

module purge
module load anaconda3
source deactivate
source activate /private/home/massran/.conda/envs/async-grad-push

# check GPU usage
nvidia-smi

mpirun python -u main.py --user-name $USER \
	--data-file-name 'qp_data.npz' \
	--graph-file-name 'erdos-renyi_n2.npz' \
	--alg 'gp' --lr 0.1 --seed 1 --num-steps 1000 \
	--log-dir '/async_maopy_playground/models/test/'
