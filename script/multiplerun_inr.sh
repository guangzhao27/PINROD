#!/bin/bash

w0_list=(20 30)
latent_list=(256)
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral

epochs=4000
dataset=$1
task_type=$2


if [ "$dataset" == "soma" ]; then
    for j in {0..0}; do
        for i in "${!w0_list[@]}" 
        do
        for k in "${!latent_list[@]}"; do
            if [ "$task_type" == "test" ]; then
                bash nersc-soma-inr-w0-low-dim.sh ${w0_list[i]} 20 ${latent_list[k]}
            else
                sbatch nersc-soma-inr-w0-low-dim.sh ${w0_list[i]} ${epochs} ${latent_list[k]} 
                echo soma
                echo ${j} ${w0_list[i]} ${latent_list[k]}
            fi
            sleep 20
        done
        done
    done
fi

# bash /pscratch/sd/g/gzhao27/INR/SOMA/script/multiplerun_inr.sh soma test
 # this line takes most memory         features_recon = func_rep.modulated_forward(coords, modulations[batch_index])