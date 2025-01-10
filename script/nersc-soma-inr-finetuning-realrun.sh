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
epochs=$5

space_factor=$6
time_factor=$7
batch_size=$8
sub_array_num=$9
trainnum=$10
missing_rate=$11
val_missing_rate=$12
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
    data.data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5' \
    data.mmap_dir='/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save' \
    inr.model_type=siren \
    data.space_factor=$space_factor \
    data.time_factor=$time_factor \
    data.ntrain=$trainnum \
    data.ntest=10 \
    optim.batch_size=$batch_size \
    optim.lr_inr=$lr \
    optim.epochs=$epochs \
    inr.latent_dim=$latent_dim \
    inr.depth=$inr_depth \
    inr.hidden_dim=$inr_hidden_dim \
    wandb.saved_checkpoint=False \
    wandb.name=$run_name \
    wandb.use_wandb=True \
    wandb.project=soma-inr \
    inr.w0=$w0 \
    data.sub_array_num=$sub_array_num \
    data.missing_rate=$missing_rate \
    data.val_missing_rate=$val_missing_rate 
    