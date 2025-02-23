#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --export=NONE
#SBATCH --cluster=tinygpu
#SBATCH --job-name=id4_base_base_gen


unset SLURM_EXPORT_ENV
module load python
conda activate l2p

export http_proxy=http://proxy:80 \n export https_proxy=http://proxy:80

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_4.255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_4.255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_1_16_255.pt' --p_task_id 0 --p_class_id 1 > trigger_0_1_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_1_16_255.pt' --p_task_id 0 --p_class_id 1 > trigger_0_1_16_255_use.txt