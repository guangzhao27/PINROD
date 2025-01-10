#!/bin/bash


cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral

epochs=2001
inr_save_name='2024-12-31SOMA-inr_w013_lr5.6e-05_depth6_hdim155-best-train-loss'
# space_factor=1
# time_factor=1
# batch_size=1
# sub_array_num=30
space_factor=1
time_factor=1
batch_size=8
sub_array_num=1
trainnum=80

task_type=$1


if [ "$task_type" == "finetune" ]; then
    for j in {0..1}; do
        sbatch /pscratch/sd/g/gzhao27/INR/SOMA/script/nersc-soma-ode-finetuning.sh ${inr_save_name} ${epochs} ${batch_size} ${trainnum} 
        echo soma
        sleep 20
    done
elif [ "$task_type" == "real-run" ]; then
    w0=$2
    lr=$3
    inr-depth=$4
    inr-hidden-dim=$5
    sbatch nersc-soma-ode.sh ${inr_save_name} ${space_factor} ${time_factor} ${batch_size} \
    6000 ${space_factor} ${time_factor} ${batch_size} ${sub_array_num} ${trainnum} ${missing_rate} ${val_missing_rate}
fi

# nersc-soma-inr-finetuning.sh run single finetunes with 20 trials