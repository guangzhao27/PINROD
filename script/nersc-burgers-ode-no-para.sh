#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g


cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
inr_save_name='2024-10-14Burgers-w0_10'
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral

lr=$1
train_num=$2
para_type=no-parameter
run_name=burgers-no-parameter_${train_num}_${lr}_0drop_6physical
python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/train.py \
    data.dataset_name=Burgers \
    dynamics.width=512 \
    dynamics.depth=3 \
    data.space_factor=10 \
    data.time_factor=10 \
    optim.epochs=10000 \
    data.seq_inter_len=20 \
    data.seq_extra_len=20 \
    optim.batch_size=16 \
    optim.lr=$lr \
    dynamics.teacher_forcing_update=10 \
    inr.save_name=$inr_save_name \
    inr.save_dir=/pscratch/sd/g/gzhao27/INR/SOMA/results \
    data.missing_rate=0.0 \
    wandb.use_wandb=True \
    wandb.project=ode_train_lr_search \
    wandb.name=${run_name} \
    para_type=$para_type \
    train_num=$train_num
    # ode_save_name=2024-10-222024-10-14Burgers-w0_10_ode.pt