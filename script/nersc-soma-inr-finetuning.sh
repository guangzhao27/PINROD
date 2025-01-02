#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g


#wandb.name is run_name also the store pt file name

w0=30
epochs=$1
latent_dim=256


sf=$2
tf=$3
bs=$4
san=$5
trainnum=$6

if ((epochs<100)); then
    run_name='test'
    echo test with factor 5 batch 1 large siren
else
    run_name=SO-fine-bs${bs}_sf${sf}_tf${tf}_san${san}_epoch${epochs}_trainnum${trainnum}
fi

conda activate coral
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/inr-finetune.py \
    data.dataset_name=SOMA \
    inr.model_type=siren \
    data.space_factor=$sf \
    data.time_factor=$tf \
    data.ntrain=$trainnum \
    data.ntest=20 \
    optim.batch_size=$bs \
    optim.lr_inr=0.001 \
    optim.epochs=$epochs \
    inr.latent_dim=$latent_dim \
    inr.depth=3 \
    inr.hidden_dim=64 \
    wandb.saved_checkpoint=False \
    wandb.name=$run_name \
    wandb.use_wandb=False \
    wandb.project=soma-inr \
    inr.w0=$w0 \
    data.sub_array_num=$san
    