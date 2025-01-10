#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=40:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
inr_save_name='2024-10-07SOMA-w0_30'

source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral

lr=$1
epochs=$2
para_type=$3
train_num=$4

p_dim=$5

if ((epochs<100)); then
    run_name='test'
    echo test
else
    run_name=soma-${para_type}_lr${lr}
fi

python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/train.py \
    data.dataset_name=SOMA \
    data.data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5' \
    data.mmap_dir='/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save' \
    dynamics.width=512 \
    dynamics.depth=3 \
    data.space_factor=2 \
    data.time_factor=2 \
    data.ntrain=80 \
    data.ntest=10 \
    optim.epochs=$epochs \
    data.seq_inter_len=20 \
    data.seq_extra_len=20 \
    optim.batch_size=8 \
    optim.lr=$lr \
    dynamics.teacher_forcing_update=10 \
    inr.save_name=$inr_save_name \
    inr.save_dir=/pscratch/sd/g/gzhao27/INR/SOMA/results \
    wandb.use_wandb=True \
    wandb.project=ode_train_soma \
    wandb.name=$run_name \
    para_type=$para_type