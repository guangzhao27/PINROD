#!/bin/bash


cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral

epochs=301
# space_factor=1
# time_factor=1
# batch_size=1
# sub_array_num=30
space_factor=1
time_factor=3
batch_size=1
sub_array_num=5
trainnum=80

task_type=$1

missing_rate=0.98
val_missing_rate=0.9


if [ "$task_type" == "finetune" ]; then
    for j in {0..1}; do
        sbatch /pscratch/sd/g/gzhao27/INR/SOMA/script/nersc-soma-inr-finetuning.sh ${epochs} ${space_factor} ${time_factor} ${batch_size} ${sub_array_num} ${trainnum} ${missing_rate} ${val_missing_rate}
        echo soma
        sleep 20
    done
elif [ "$task_type" == "real-run" ]; then
    w0=$2
    lr=$3
    inr-depth=$4
    inr-hidden-dim=$5
    sbatch nersc-soma-inr-finetuning-realrun.sh ${w0} ${lr} ${inr-depth} ${inr-hidden-dim} \
    4000 ${space_factor} ${time_factor} ${batch_size} ${sub_array_num} ${trainnum} ${missing_rate} ${val_missing_rate}
fi

# nersc-soma-inr-finetuning.sh run single finetunes with 20 trials