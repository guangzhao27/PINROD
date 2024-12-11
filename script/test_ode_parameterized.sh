#!/bin/bash
# para_type: concatenate modulation
# forcing_update_list=(20)
train_num_list=(200)
lr_list=('1e-5')
para_type_list=(concatenate concatenate_test)
p_encoding_dim_list=(32 64 128)
# log_p_list=(true false)
# p_activation=(relu swish)
cd /pscratch/sd/g/gzhao27/INR/SOMA/script/
source ~/anaconda3/etc/profile.d/conda.sh
conda activate coral
epochs=20
para_type=modulation
for a in ${!p_encoding_dim_list[@]}; do
for k in ${!para_type_list[@]}; do
for j in ${!train_num_list[@]}; do
    for i in "${!lr_list[@]}" 
    do
    
        bash nersc-burgers-ode-parameterized.sh ${lr_list[i]} ${train_num_list[j]} ${para_type_list[k]} ${p_encoding_dim_list[a]} ${epochs}
        echo ${lr_list[i]} ${train_num_list[j]}
    done
done
done
done