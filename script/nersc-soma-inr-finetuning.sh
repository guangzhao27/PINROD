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
missing_rate=$7
val_missing_rate=$8

if ((epochs<100)); then
    run_name='test'
    echo test with factor 5 batch 1 large siren
else
    run_name=SO-fine-bs${bs}_sf${sf}_tf${tf}_san${san}_epoch${epochs}_trainnum${trainnum}_mr${missing_rate}_vmr${val_missing_rate}
fi

conda activate coral
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/inr-finetune.py \
    data.dataset_name=SOMA \
    data.data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5' \
    data.mmap_dir='/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save' \
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
    data.sub_array_num=$san \
    data.missing_rate=$missing_rate \
    data.val_missing_rate=$val_missing_rate 
    