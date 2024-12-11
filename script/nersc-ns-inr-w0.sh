#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g

w0=$1

conda activate coral
python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/inr.py \
    data.dataset_name=NS \
    inr.model_type=siren \
    data.space_factor=2 \
    data.ntrain=1000 \
    data.ntest=100 \
    data.missing_rate=0.5 \
    optim.batch_size=8 \
    optim.lr_inr=0.001 \
    optim.epochs=1000 \
    inr.latent_dim=128 \
    inr.depth=3 \
    inr.hidden_dim=32 \
    wandb.saved_checkpoint=False \
    wandb.name=NS-w0_$w0 \
    wandb.use_wandb=True \
    wandb.project=w0search \
    inr.w0=$w0
    