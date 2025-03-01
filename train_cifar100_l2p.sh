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

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_5_0_16_255.pt' --p_task_id 5 --p_class_id 0 > trigger_5_0_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_5_0_16_255.pt' --p_task_id 5 --p_class_id 0 > trigger_5_0_16_255_use.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_16_255_bestselect.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_16_255_bestselect.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_16_255_retrain.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_16_255_retrain.txt


python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'test.pt' --p_task_id 0 --p_class_id 0 --retrain False --best_select False > test



python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.005 --trigger_path 'trigger_0_0_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.1 --trigger_path 'trigger_0_0_0.1_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_0.1_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./bestselect --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_0.1_16_255.pt' --best_select True --p_task_id 0 --p_class_id 0 > trigger_0_0_0.1_16_255_bestselect.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./retrain --use_trigger False --poison_rate 0.1 --trigger_path 'trigger_0_0_16_255_retrain.pt' --retrain True --p_task_id 0 --p_class_id 0 > trigger_0_0_16_255_retrain.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./retrain --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_16_255_retrain.pt' --retrain True --p_task_id 0 --p_class_id 0 > trigger_0_0_16_255_retrain.txt





python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.1 --trigger_path 'trigger_0_0_0.1_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_0.1_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_0.1_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_0.1_16_255_use.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.05 --trigger_path 'trigger_0_0_0.05_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_0.05_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.05 --trigger_path 'trigger_0_0_0.05_16_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_0.05_16_255_use.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.1 --trigger_path 'trigger_0_0_0.1_4_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_0.1_4_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_0_0_0.1_4_255.pt' --p_task_id 0 --p_class_id 0 > trigger_0_0_0.1_4_255_use.txt


python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.1 --trigger_path 'trigger_9_0_0.1_16_255.pt' --p_task_id 9 --p_class_id 0 > trigger_9_0_0.1_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path 'trigger_9_0_0.1_16_255.pt' --p_task_id 9 --p_class_id 0 > trigger_9_0_0.1_16_255_use.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.1 --black_box True --trigger_path 'trigger_9_0_0.1_16_255_black.pt' --p_task_id 9 --p_class_id 0 > trigger_9_0_0.1_16_255_black.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --black_box True --trigger_path 'trigger_9_0_0.1_16_255_black.pt' --p_task_id 9 --p_class_id 0 > trigger_9_0_0.1_16_255_black_use.txt


python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger False --poison_rate 0.1 --trigger_path '5_0_0_0.1_16_255.pt' --p_task_id 0 --p_class_id 0 > 5_0_0_0.1_16_255.txt

python main.py cifar100_l2p --model vit_base_patch16_224 --batch-size 16 --data-path ./../local_datasets/ --output_dir ./output --use_trigger True --poison_rate 0.1 --trigger_path '5_0_0_0.1_16_255.pt' --p_task_id 0 --p_class_id 0 > 5_0_0_0.1_16_255_use.txt










