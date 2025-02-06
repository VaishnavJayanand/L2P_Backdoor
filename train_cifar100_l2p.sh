#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --export=NONE
#SBATCH --cluster=tinygpu
#SBATCH --job-name=trigger_0_0.1_vit_base_patch16_224_bestselect



unset SLURM_EXPORT_ENV
module load python
conda activate l2p

export http_proxy=http://proxy:80 \n export https_proxy=http://proxy:80

/home/woody/iwi1/iwi1102h/software/private/conda/envs/l2p/bin/python -m torch.distributed.launch \
        --nproc_per_node=1 \
        --use_env main.py \
        cifar100_l2p \
        --model vit_base_patch16_224 \
        --batch-size 16 \
        --data-path ./local_datasets/ \
        --output_dir ./output \
        --use_trigger false \
        --poison_rate 0.1 \
        --epochs 5 \
        --trigger_path '/home/woody/iwi1/iwi1102h/trigger/trigger_0_0.1_vit_base_patch16_224.pt' \
        --p_task_id 0

