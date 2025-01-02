#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g


#wandb.name is run_name also the store pt file name

w0=$1
lr=$2
inr_depth=$3
inr_hidden_dim=$4
epochs=4000
latent_dim=256


if ((epochs<100)); then
    run_name='test'
    echo test with factor 5 batch 1
else
    run_name=SOMA-inr_w0${w0}_lr${lr}_depth${inr_depth}_hdim${inr_hidden_dim}
fi

conda activate coral
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/inr.py \
    data.dataset_name=SOMA \
    inr.model_type=siren \
    data.space_factor=4 \
    data.time_factor=2 \
    data.ntrain=80 \
    data.ntest=20 \
    optim.batch_size=2 \
    optim.lr_inr=$lr \
    optim.epochs=$epochs \
    inr.latent_dim=$latent_dim \
    inr.depth=$inr_depth \
    inr.hidden_dim=$inr_hidden_dim \
    wandb.saved_checkpoint=False \
    wandb.name=$run_name \
    wandb.use_wandb=True \
    wandb.project=soma-inr \
    inr.w0=$w0
    