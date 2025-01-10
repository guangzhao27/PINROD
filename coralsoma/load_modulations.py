import os
from pathlib import Path
from torchdiffeq import odeint
import einops
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricLoader
from coral.metalearning import graph_outer_step as outer_step
import sys
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append(str(Path(__file__).parents[1]))
from coral.utils.models.scheduling import ode_scheduling
from utils.data.unstructure_dataset import GraphNavierStokes, collate_graph_inr, GraphSomaDataset
from torch_geometric.data import Data
from torch.utils.data import Dataset

def load_soma_graph_modulations_each_frame(
    inr_save_name, 
    inr, 
    trainset, 
    type, 
    device,
    inner_steps,
    alpha,
    
):
    N = len(trainset)
    latent_dim = trainset.latent_dim
    time_factor = trainset.time_factor
    space_factor = trainset.space_factor
    T = trainset.T
    
    modulation_path = os.path.join(
        '/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_modulation',
        inr_save_name + f'num{N}ld{latent_dim}tf{time_factor}sf{space_factor}-{type}.pt'
        )
    
    if not os.path.exists(modulation_path):
        modulations = torch.zeros(N, T, latent_dim)
        pde_tensor = torch.zeros(N, 1)
        for i in range(N):
            graph = trainset[i]
            for t in range(T):
                graph0 = Data()
                booltensor = graph.time == t
                graph0.images = graph.feat[booltensor].to(device)
                graph0.pos = graph.space_emb[booltensor].to(device)
                graph0.batch = (graph.time[booltensor] - graph.time[booltensor].min()).to(device) 
                graph0.modulations = graph.latent_vector[[t]].to(device)
                
                outputs = outer_step(
                    inr, 
                    graph0, 
                    inner_steps, 
                    alpha, 
                    is_train=False, 
                    return_reconstructions=False, 
                    gradient_checkpointing=True,
                    use_rel_loss=False,
                    loss_type="mse",
                )
                
                z0 = outputs["modulations"].cpu().detach()
                modulations[i, t] = z0
            pde_tensor[i] = graph.pde_parameter
        torch.save(
            {
                'modulations': modulations, 
                'pde_tensor': pde_tensor,
            },
            modulation_path)
    else:
        save_data = torch.load(modulation_path)
        modulations = save_data['modulations']
        pde_tensor = save_data['pde_tensor']
    
    # properties_to_keep = ['latent_vector', 'pde_parameter'] 
    # for i in range(N):
    #     # trainset.update_latent_vector(i, modulations[i])
    #     graph = trainset[i]
    #     pde_tensor[i] = graph.pde_parameter
        # for key in list(graph.keys()):
        #     if key not in properties_to_keep:
        #         del graph[key]
                
    train_modulation_set = ModulationDataset(modulations, pde_tensor)
    
    z_mean = modulations.mean().item()
    z_std = modulations.std().item()
    return z_mean, z_std, train_modulation_set
            

def load_graph_modulations(
    trainset,
    inr,
    inner_steps=3,
    alpha=0.01,
    batch_size=8,
    device= None,
):
    #pre process all training data
    # update encoded inr latent vector to the model
    """WARNING : This function assumes that we can encode a full trajectory"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, collate_fn=collate_graph_inr)
        
    fit_train_mse = 0
    dataset_idx = 0
    z_list = []
        
    for substep, graph in enumerate(train_loader):
        # torch.cuda.empty_cache()
        inr.train()
        n_samples = len(graph)

        #modify for outer_step function
        graph.images = graph.feat
        graph.pos = graph.space_emb
        graph.batch = graph.time
        graph.modulations = graph.latent_vector
        graph.to(device)
        # with torch.no_grad():
        outputs = outer_step(
            inr,
            graph,
            inner_steps,
            alpha,
            is_train=False,
            return_reconstructions=False,
            gradient_checkpointing=False,
            use_rel_loss=False,
            loss_type="mse",
        )
        

        loss = outputs["loss"].cpu().detach()
        fit_train_mse += loss.item() * n_samples
        
        
        z0 = outputs["modulations"].cpu().detach()
        
        # modify dataset latent vector values
        latent_idx = 0
        for i in range(dataset_idx, dataset_idx+len(graph)):
            tempT = trainset[i].T
            if getattr(trainset, 'name', None) == 'SOMA':
                trainset.update_latent_vector(i, z0[latent_idx: latent_idx+tempT])
            else:
                trainset[i].latent_vector = z0[latent_idx: latent_idx+tempT]
            latent_idx+=tempT
        dataset_idx += len(graph)
        
        z_list.append(z0)
        # graph.latent_vector = z0
    
    ntrain = len(trainset)
    train_loss = fit_train_mse / ntrain
    z_tensor = torch.cat(z_list, dim=0)
    z_mean = z_tensor.mean().item()
    z_std = z_tensor.std().item()
    return z_mean, z_std

def graph_ode_inr_predict(model, inr, graph, timestamps=None, z_transform=None, z_invtransform=None, parameterized=False, device=None):
    # timstamps orders:
    # 1. use explicit defined timestamps. 2. use time_embedding defined in dataset. 3. use default timestamps generate by nval
    from train import ode_pred_z, modulation_fix
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = graph.to(device)
    images = graph.feat.to(device)
    coords = graph.space_emb.to(device)
    batch = graph.time
    
    n_samples = len(graph)
    
    T = graph.T[0]
    
    modulations = graph.latent_vector.to(device)
    latent_dim = modulations.size(-1)
    
    # TODO: break the batch into small batch to reduce memory requirements
    inr_pred = inr.modulated_forward(coords, modulations[batch])
    inr_pred_loss = ((images - inr_pred)**2).mean()
    
    # modulations = modulations.reshape(n_samples, T, latent_dim)
    # modulations = modulations.permute(0, 2, 1)
    
    # if z_transform is not None:
    #     modulations = z_transform(modulations)
    modulations = modulation_fix(modulations, n_samples, T, latent_dim, z_transform, device=device)
    if timestamps is not None:
        timestamps = timestamps.to(device)
    elif hasattr(graph, 'time_emb'):
        timestamps = graph[0].time_emb.to(device)
    else:
        timestamps = torch.arange(T).float().to(device)
        print("This feature should be remove")
        
    
    z_pred = ode_pred_z(model, graph, modulations, timestamps, 0, parameterized)
    # z_pred = ode_scheduling(
    #                 odeint, model, modulations, timestamps, epsilon=0
    #             )
    z_pred_loss = ((z_pred - modulations) ** 2).mean()

    if z_invtransform is not None:
        z_pred = z_invtransform(z_pred)
    z_pred = z_pred.permute(0, 2, 1)
    z_pred = z_pred.reshape(n_samples*T, latent_dim)
    
    # TODO: break the batch into small batch to reduce memory requirements
    ode_pred = inr.modulated_forward(coords, z_pred[batch])
    ode_pred_loss = ((images - ode_pred)**2).mean()
    
    # modulations = modulations.permute(0, 2, 1)
    # modulations = modulations.reshape(n_samples*T, latent_dim)
    # inr_pred = inr.modulated_forward(coords, modulations[batch])
    # inr_pred_loss = ((images - inr_pred)**2).mean()
    
    # ode_pred_loss_total = n_samples*ode_pred_loss
    # z_pred_loss_total = n_samples*z_pred_loss
    
    outputs = {
        'ode_pred':ode_pred.detach().cpu(),
        'ode_pred_loss': ode_pred_loss,
        'inr_pred':inr_pred.detach().cpu(),
        'inr_pred_loss': inr_pred_loss,
        'z_pred_loss': z_pred_loss,
    }
    
    return outputs


class ModulationDataset(Dataset):
    
    def __init__(self, modulations, pde_tensor):
        self.modulations = modulations 
        self.pde_tensor = pde_tensor
        
    def __len__(self):
        return self.modulations.size(0)
    
    def __getitem__(self, idx):
        return Data(latent_vector=self.modulations[idx], pde_parameter=self.pde_tensor[idx])
        