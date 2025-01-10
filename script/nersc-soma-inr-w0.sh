#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g


#wandb.name is run_name also the store pt file name

w0=$1
epochs=$2
latent_dim=$3

if ((epochs<100)); then
    run_name='test'
    echo test original resolution 
else
    run_name=SOMA-inr_w0${w0}_ld${latent_dim}_large_HR
fi

conda activate coral
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/inr.py \
    data.dataset_name=SOMA \
    data.data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5' \
    data.mmap_dir='/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save' \
    inr.model_type=siren \
    data.space_factor=4 \
    data.time_factor=1 \
    data.ntrain=10 \
    data.ntest=20 \
    optim.batch_size=1 \
    optim.lr_inr=0.001 \
    optim.epochs=$epochs \
    inr.latent_dim=$latent_dim \
    inr.depth=6 \
    inr.hidden_dim=256 \
    wandb.saved_checkpoint=False \
    wandb.name=$run_name \
    wandb.use_wandb=True \
    wandb.project=soma-inr \
    inr.w0=$w0 \
    data.sub_array_num=2
    