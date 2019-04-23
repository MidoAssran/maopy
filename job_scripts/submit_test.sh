#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/checkpoint/%u/async_maopy_playground/test.out
#SBATCH --error=/checkpoint/%u/async_maopy_playground/test.err
#SBATCH --nodes=4
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --time=00:10:00
#SBATCH --partition=dev

module purge
module load anaconda3
module load openmpi
source deactivate
source activate /private/home/massran/.conda/envs/agp

mpirun --report-bindings --oversubscribe --bind-to hwthread \
	--mca btl_base_warn_component_unused 0 \
	--mca btl_openib_warn_default_gid_prefix 0 \
	python -u main.py  \
	--data-file-name 'qp_data_sg.npz' \
	--graph-file-name 'erdos-renyi_n8.npz' \
	--alg 'gp' --lr 0.0001 --seed 1 --num-steps 10 \
	--log-dir '/async_maopy_playground/qp/test/' \
	--asynch
