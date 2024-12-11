#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g


conda activate coral
python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/train.py \
    data.dataset_name=NS \
    dynamics.width=512 \
    dynamics.depth=3 \
    data.space_factor=2 \
    data.time_factor=2 \
    optim.epochs=10000 \
    data.seq_inter_len=20 \
    data.seq_extra_len=20 \
    optim.batch_size=64 \
    optim.lr=0.001 \
    dynamics.teacher_forcing_update=10 \
    inr.save_name=NS_test_show \
    inr.save_dir=/pscratch/sd/g/gzhao27/INR/SOMA/results \
    data.missing_rate=0.1 \
    wandb.use_wandb=True \
    wandb.project=ode_train \
    wandb.name=ns-no-parameter