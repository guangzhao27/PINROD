#!/bin/bash


cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral

epochs=601
# space_factor=1
# time_factor=1
# batch_size=1
# sub_array_num=30
space_factor=4
time_factor=2
batch_size=2
sub_array_num=1
trainnum=80

task_type=$1


if [ "$task_type" == "finetune" ]; then
    for j in {0..1}; do
        sbatch /pscratch/sd/g/gzhao27/INR/SOMA/script/nersc-soma-inr-finetuning.sh ${epochs} ${space_factor} ${time_factor} ${batch_size} ${sub_array_num} ${trainnum}
        echo soma
        sleep 20
    done
elif [ "$task_type" == "real-run" ]; then
    w0=$2
    lr=$3
    inr-depth=$4
    inr-hidden-dim=$5
    sbatch nersc-soma-inr-finetuning-realrun.sh ${w0} ${lr} ${inr-depth} ${inr-hidden-dim}
fi

# nersc-soma-inr-finetuning.sh run single finetunes with 20 trials