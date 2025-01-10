#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=40:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=gpu
#SBATCH --gpus-per-node=1
#SBATCH --account=m4259_g
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/

source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral

#these values will be decide by search algorithm
lr=1e-6
para_type=concatenate
p_dim=32

width=512
depth=3

inr_save_name=$1
epochs=$2
bs=$3
trainnum=$4

if ((epochs<100)); then
    run_name='test'
    echo test
else
    run_name=SO-ODE-fine-${inr_save_name}_epoch${epochs}_trainnum${trainnum}_bs${bs}
fi

python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/ode_finetune.py \
    data.dataset_name=SOMA \
    data.data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5' \
    data.mmap_dir='/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save' \
    dynamics.width=$width \
    dynamics.depth=$depth \
    data.space_factor=1 \
    data.time_factor=1 \
    data.ntrain=$trainnum \
    data.ntest=10 \
    optim.epochs=$epochs \
    data.seq_inter_len=20 \
    data.seq_extra_len=20 \
    optim.batch_size=$bs \
    optim.lr=$lr \
    dynamics.teacher_forcing_update=10 \
    inr.save_name=$inr_save_name \
    inr.save_dir=/pscratch/sd/g/gzhao27/INR/SOMA/results \
    wandb.use_wandb=False \
    wandb.project=ode_train_soma \
    wandb.name=$run_name \
    para_type=$para_type \
    p_dim=$p_dim