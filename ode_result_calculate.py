import wandb
import torch.nn as nn
import torch
import numpy as np
import hydra
import einops
import os
import sys
from pathlib import Path
from dynamics_modeling.eval import batch_eval_loop
from torch.utils.data import DataLoader


os.environ["WANDB_DIR"] = '/pscratch/sd/g/gzhao27/INR/coral/wandb'
os.environ["RESULTS_DIR"] = ''

# sys.path.append(str(Path(__file__).parents[1]))
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append('/pscratch/sd/g/gzhao27/INR/SOMA')
from coral.utils.plot import show
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.load_inr import create_inr_instance, load_inr_model
from coral.utils.models.get_inr_reconstructions import get_reconstructions
from coralsoma.load_modulations import load_graph_modulations, graph_ode_inr_predict
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import (KEY_TO_INDEX, TemporalDatasetWithCode)
from coral.mlp import Derivative
from torchdiffeq import odeint
from omegaconf import DictConfig, OmegaConf
from utils.data.unstructure_dataset import GraphNavierStokes, collate_graph_inr, GraphSomaDataset
from utils.data.unstructure_dataset import GraphBurgers
# with hydra.initialize(config_path="../config/"):
#     # Compose the configuration (equivalent to loading ode.yaml)
#     cfg = hydra.compose(config_name="ode.yaml")

dataset_name="Burgers"


if dataset_name == "Burgers":
    
    # inr
    inr_save_name = "2024-10-14Burgers-w0_10"
    inr_save_dir = "/pscratch/sd/g/gzhao27/INR/SOMA/results"
    ode_save_name = "2024-10-242024-10-14Burgers-w0_101e-05_40_ode.pt"
    ode_save_path = os.path.join(inr_save_dir, ode_save_name)
    ode_results = torch.load(ode_save_path)
    parameterized = False
    cfg = ode_results['cfg']
    
    input_dim = 1
    output_dim = 1
    

torch.set_default_dtype(torch.float32)
data_dir = cfg.data.dir
dataset_name = cfg.data.dataset_name
ntrain = cfg.data.ntrain
ntest = cfg.data.ntest
data_to_encode = cfg.data.data_to_encode
space_factor = cfg.data.space_factor
time_factor = cfg.data.time_factor
seed = cfg.data.seed
same_grid = cfg.data.same_grid
seq_inter_len = cfg.data.seq_inter_len
seq_extra_len = cfg.data.seq_extra_len

missing_rate = cfg.data.missing_rate

inner_steps = cfg.inr.inner_steps

# dynamics
model_type = cfg.dynamics.model_type
hidden = cfg.dynamics.width
depth = cfg.dynamics.depth
epsilon = cfg.dynamics.teacher_forcing_init
epsilon_t = cfg.dynamics.teacher_forcing_decay
epsilon_freq = cfg.dynamics.teacher_forcing_update





if inr_save_name is not None:
    multichannel = False
    tmp = torch.load(os.path.join(inr_save_dir, inr_save_name+'.pt'))
    latent_dim = tmp["cfg"].inr.latent_dim
    # print('fix sub_tr to space_factor')
    space_factor = tmp["cfg"].data.space_factor
    seed = tmp["cfg"].data.seed
    # missing_rate = tmp["cfg"].data.missing_rate
    
    inr, alpha = load_inr_model(
        inr_save_dir,
        inr_save_name,
        data_to_encode,
        input_dim=input_dim,
        output_dim=output_dim,
    )
    
z_mean = ode_results['z_mean']
z_std = ode_results['z_std']
z_transform = lambda tensor: (tensor - z_mean) / z_std
z_invtransform = lambda tensor: (tensor * z_std) + z_mean

model_state_dict = ode_results['ode']
model = Derivative(output_dim, latent_dim, hidden, depth).cuda()
model.load_state_dict(model_state_dict)

if dataset_name=='Burgers':
    test_space_factor = 1
    test_time_factor = 1
    
    X = (1024+test_space_factor-1)//test_space_factor
    T = (201+test_time_factor-1)//test_time_factor

    trainnum = 100
    valnum = 100
    testnum = 100
    train_p_list=(0.001, 0.004, 0.02)
    val_p_list = (0.01, )
    test_p_list = (0.01, )

    path_format = "/pscratch/sd/g/gzhao27/INR/data/1D_Burgers_Sols_Nu{}.hdf5"
    def create_dict(p_list, start, Dnum):
        Ddict = {}
        for p in p_list:
            temppath = path_format.format(p)
            Ddict[temppath] = (list(range(start, start+Dnum)), p)
        return Ddict

    traindict = create_dict(train_p_list, 0, trainnum)
    valdict = create_dict(val_p_list, trainnum, valnum)
    testdict = create_dict(test_p_list, trainnum+valnum, testnum)
        

    testset = GraphBurgers(
        datapath_dict=testdict,
        latent_dim=128,
        missing_rate=0.,
        space_factor=test_space_factor,
        time_factor=test_time_factor,
    )


    device = torch.device('cuda')
    alpha = alpha.to(device)
    model.to(device)
    inr.to(device)
    load_graph_modulations(
        testset,
        inr,
        inner_steps=inner_steps,
        alpha=alpha,
        batch_size=2,
    )

batch_size = 10

test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_graph_inr,
    num_workers=1,
)

sys.path.append('/pscratch/sd/g/gzhao27/INR/SOMA/coralsoma')
from train import modulation_fix, ode_pred_z

# TODO: for graph in test_loader
T_test = testset[0].T
ntest = len(testset)
dt=1
timestamps_test = torch.arange(0, T_test, dt).float().to(device)
timestamps_test /= time_factor
image_test_mse = 0
z_test_mse = 0

for substep, graph in enumerate(test_loader): 
    
    model.eval()
    n_samples = len(graph)
    modulations = graph.latent_vector

    modulations = modulation_fix(modulations, n_samples, T_test, latent_dim, z_transform)


    z_pred = ode_pred_z(model, graph, modulations, timestamps_test, 0, False)
    loss = ((z_pred - modulations) ** 2).mean()
    z_test_mse  += loss.item()*n_samples

    outputs = graph_ode_inr_predict(
        model, 
        inr, 
        graph, 
        timestamps=timestamps_test,
        z_transform=z_transform,
        z_invtransform=z_invtransform
        )
    image_pred_loss = outputs['ode_pred_loss']
    image_test_mse += image_pred_loss * n_samples
    if substep == 0:
        ode_pred = outputs['ode_pred']
        inr_pred = outputs['inr_pred']

image_test_mse = image_test_mse/ntest
z_test_mse = z_test_mse/ntest
print("val latent loss:", z_test_mse)
print("val image loss:", image_test_mse)

save_path = ode_save_name+'test.pt'
torch.save(
    {
        "testdata": testset,
        "inr_pred": inr_pred.cpu(),
        "ode_pred": ode_pred.cpu(),
        },
    save_path
)

