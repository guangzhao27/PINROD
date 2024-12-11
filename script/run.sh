conda activate /pscratch/sd/g/gzhao27/conda/coral

# coral dynamic inr training file: /pscratch/sd/g/gzhao27/INR/coral/inr/inr.py
#       ODE training file: /pscratch/sd/g/gzhao27/INR/coral/dynamics_modeling/train.py

# Information about ODE training:
# ODE training file: /pscratch/sd/g/gzhao27/INR/coral/dynamics_modeling/train.py
# ODE function: /pscratch/sd/g/gzhao27/INR/coral/coral/utils/models/scheduling.py
# https://vscode.dev/github/LouisSerrano/coral/blob/main/coral/utils/models/scheduling.py#L9 in this line, what is epsilon doing?

python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/inr.py \
    data.dataset_name=NS \
    inr.model_type=siren \
    data.sub_from=2 \
    data.space_factor=2 \
    optim.batch_size=8 \
    optim.lr_inr=0.001 \
    optim.epochs=100 \
    inr.latent_dim=128 \
    inr.depth=3 \
    inr.hidden_dim=32 \
    wandb.saved_checkpoint=False \
    data.missing_rate=0.1 \
    wandb.name=NS-test \
    data.ntrain=1000 \
    data.ntest=100

python /pscratch/sd/g/gzhao27/INR/SOMA/coralsoma/train.py \
    data.dataset_name=NS \
    dynamics.width=512 \
    dynamics.depth=3 \
    data.space_factor=2 \
    data.time_factor=2 \
    optim.epochs=10000 \
    data.seq_inter_len=20 \
    data.seq_extra_len=20 \
    optim.batch_size=64 \
    optim.lr=0.001 \
    dynamics.teacher_forcing_update=10 \
    inr.save_name=NS_test \
    inr.save_dir=/pscratch/sd/g/gzhao27/INR/SOMA/results \
    data.missing_rate=0.1 \
    wandb.use_wandb=True