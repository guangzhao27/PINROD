#!/bin/bash
# para_type: concatenate modulation
# forcing_update_list=(20)
train_num_list=(200)
lr_list=(0.001 0.0001 '1e-5')
para_type_list=(no-parameter)
p_encoding_dim_list=(128)
log_p_list=(true false)
# p_activation=(relu swish)
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral
epochs=10000
dataset=$1
task_type=$2

if [ "$dataset" == "burgers" ]; then
    for a in ${!p_encoding_dim_list[@]}; do
    for k in ${!para_type_list[@]}; do
    for j in ${!train_num_list[@]}; do
        for i in "${!lr_list[@]}" 
        do
            if [ "$task_type" == "test" ]; then
                bash nersc-burgers-ode-parameterized.sh ${lr_list[i]} ${train_num_list[j]} ${para_type_list[k]} ${p_encoding_dim_list[a]} 20
            else
                sbatch nersc-burgers-ode-parameterized.sh ${lr_list[i]} ${train_num_list[j]} ${para_type_list[k]} ${p_encoding_dim_list[a]} ${epochs}
            fi
            echo ${lr_list[i]} ${train_num_list[j]}
            sleep 20
        done
    done
    done
    done

elif [ "$dataset" == "soma" ]; then
    for k in ${!para_type_list[@]}; do
        for i in "${!lr_list[@]}"
        do
            echo ${lr_list[i]} ${task_type}
            if [ "$task_type" == "test" ]; then
                bash nersc-soma-ode.sh ${lr_list[i]} 20 ${para_type_list[k]}
            else
                sbatch nersc-soma-ode.sh ${lr_list[i]} ${epochs} ${para_type_list[k]}
                echo soma
            fi
            
            sleep 20
        done
    done
fi
