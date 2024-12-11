#!/bin/bash

# forcing_update_list=(20)
# train_num_list=(100 1000)
# lr_list=(0.001 0.0001 '1e-5')
train_num_list=(200)
lr_list=('1e-5')
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral
for j in ${!train_num_list[@]}; do
    for i in "${!lr_list[@]}" 
    do
    
        sbatch nersc-burgers-ode-no-para.sh ${lr_list[i]} ${train_num_list[j]}
        echo ${lr_list[i]} ${train_num_list[j]}
        sleep 20
    done
done

