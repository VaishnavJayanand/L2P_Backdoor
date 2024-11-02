#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --cluster=tinygpu
#SBATCH --job-name=l2p_simple_backdoor_30_.7

unset SLURM_EXPORT_ENV
module load python
source activate l2p
python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        cifar100_l2p \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path ./local_datasets/ \
        --output_dir ./output 
