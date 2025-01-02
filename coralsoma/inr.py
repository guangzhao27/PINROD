import os
import sys
from pathlib import Path
from pickletools import OpcodeInfo
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
sys.path.append(str(Path(__file__).parents[1]))
print(sys.executable)
import einops
import hydra
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from omegaconf import DictConfig, OmegaConf
from utils.data.unstructure_dataset import (GraphNavierStokes, 
                                            collate_graph_inr, 
                                            GraphSomaDataset, 
                                            GraphBurgers, 
                                            create_burgers_dataset, 
                                            )


from coral.losses import batch_mse_rel_fn
from coral.metalearning import graph_outer_step as outer_step
from coral.mlp import MLP, Derivative, ResNet
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, rearrange
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.plot import show
from datetime import datetime
from time import time
from torch_geometric.data import Data



os.environ["WANDB_DIR"] = '/pscratch/sd/g/gzhao27/INR/coral/wandb'
os.environ["RESULTS_DIR"] = ''


def divide_array_indexes(N, k):
    # Base size and extra elements
    base_size = N // k
    extra = N % k
    
    # Create the index ranges
    indexes = []
    start = 0
    for i in range(k):
        end = start + base_size + (1 if i < extra else 0)
        indexes.append(torch.tensor(list(range(start, end))))
        start = end
    
    return indexes

@hydra.main(config_path="../config/", config_name="siren.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)
    current_date_str = datetime.now().strftime("%Y-%m-%d")
    # submitit.JobEnvironment()
    # data
    saved_checkpoint = cfg.wandb.saved_checkpoint
    if saved_checkpoint:
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name
        checkpoint = torch.load(cfg.wandb.checkpoint_path)
        cfg = checkpoint['cfg']
    elif saved_checkpoint == False:
        #wandb
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name

    #data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    mmap_dir = cfg.data.mmap_dir
    data_to_encode = cfg.data.data_to_encode
    # sub_from = cfg.data.sub_from
    space_factor = cfg.data.space_factor
    time_factor = cfg.data.time_factor
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len
    missing_rate = cfg.data.missing_rate
    sub_array_num = cfg.data.sub_array_num

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr_inr = cfg.optim.lr_inr
    gamma_step = cfg.optim.gamma_step
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs

    # inr
    model_type = cfg.inr.model_type
    latent_dim = cfg.inr.latent_dim

    # wandb
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"),
                     f"wandb/{cfg.wandb.dir}/{dataset_name}")
        if cfg.wandb.dir is not None
        else None
    )

    sweep_id = cfg.wandb.sweep_id
    device = torch.device("cuda")
    print("run dir given", run_dir)

    if cfg.wandb.use_wandb:
        run = wandb.init(
            #entity=entity,
            project=project,
            name=run_name,
            id=run_id,
            dir=run_dir,
            # resume='allow',
        )
        if run_dir is not None:
            os.symlink(run.dir.split("/files")[0], run_dir)

        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        run_name = wandb.run.name

        print("id", run.id)
        print("dir", run.dir)

        if data_to_encode is not None:
            RESULTS_DIR = (
                Path(os.getenv("WANDB_DIR")) / dataset_name / data_to_encode / "inr"
            )
        else:
            RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"

        os.makedirs(str(RESULTS_DIR), exist_ok=True)
    
    set_seed(seed)
    
    # feature transform
    feat_transform, feat_inv_transform = None, None

    if dataset_name == "NS":
        input_dim=2
        output_dim=1
        trainset = GraphNavierStokes(split='train', ssub=space_factor, datanum=cfg.data.ntrain, missing_rate=missing_rate, latent_dim=latent_dim)
        valset = GraphNavierStokes(split='val', ssub=space_factor, trainnum=cfg.data.ntrain, datanum=cfg.data.ntest, missing_rate=missing_rate, latent_dim=latent_dim)
        testset = GraphNavierStokes(split='test', datanum=200, missing_rate=missing_rate, latent_dim=latent_dim)
    elif dataset_name == "SOMA":
        # mmap_dir = '/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save'
        raw_np_dir = '/pscratch/sd/g/gzhao27/INR/SOMA/results/impliciBottomDrag_np'
        input_dim = 3
        output_dim = 1
        feature_set = [10]
        trainset = GraphSomaDataset(
            data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
            train_num=ntrain, 
            feature_set=[10],
            space_factor=space_factor,
            time_factor=time_factor, 
            latent_dim=latent_dim,
            mmap_dir=mmap_dir,
        )
        feat_transform, feat_inv_transform = trainset.create_normalize_from_dataset()
        trainset.update_feat_transform(feat_transform)
        valset = GraphSomaDataset(
            data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
            train_num=10, 
            feature_set=[10],
            space_factor=space_factor,
            time_factor=time_factor, 
            latent_dim=latent_dim,
            split='val',
            feature_transform=feat_transform,
            mmap_dir=mmap_dir,
        )
        testset = GraphSomaDataset(
            data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
            train_num=10, 
            feature_set=[10],
            space_factor=space_factor,
            time_factor=time_factor, 
            latent_dim=latent_dim,
            split='test',
            feature_transform=feat_transform,
            mmap_dir=mmap_dir,
        )
    elif dataset_name == "Burgers":
        input_dim = 1
        output_dim = 1
        trainset, valset, testset, p_mean, p_std = create_burgers_dataset()
          
    else:
        raise NotImplementedError(f"The dataset ${dataset_name} does not have a corresponding class.")

    

    ntrain = len(trainset)
    nval = len(valset)
    ntest = len(testset)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_inr)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_inr)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_inr)

    print("train", len(trainset))
    print("val", len(valset))



    inr = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device=device
    )

    alpha = nn.Parameter(torch.Tensor([lr_code]).to(device))
    meta_lr_code = meta_lr_code
    weight_decay_lr_code = weight_decay_code

    optimizer = torch.optim.AdamW(
        [
            {"params": inr.parameters(), "lr": lr_inr},
            {"params": alpha, "lr": meta_lr_code, "weight_decay": weight_decay_lr_code},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    if saved_checkpoint:
        inr.load_state_dict(checkpoint['inr'])
        optimizer.load_state_dict(checkpoint['optimizer_inr']) 
        epoch_start = checkpoint['epoch']
        alpha = checkpoint['alpha']
        best_loss = checkpoint['loss']
        cfg = checkpoint['cfg']
        print("epoch_start, alpha, best_loss", epoch_start, alpha.item(), best_loss)
        print("cfg : ", cfg)
    elif saved_checkpoint == False:
        epoch_start = 0
        best_loss = np.inf

    if cfg.wandb.use_wandb:
        wandb.log({"results_dir": str(RESULTS_DIR)}, step=epoch_start, commit=False)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=gamma_step,
        patience=500,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08,
        verbose=True,
    )

    for step in range(epoch_start, epochs):
        rel_train_mse = 0
        rel_test_mse = 0
        fit_train_mse = 0
        fit_test_mse = 0
        use_rel_loss = step % 10 == 0
        step_show = step % 10 == 0
        step_show_last = step == epochs - 1
        start = time()
        for substep, graph in enumerate(train_loader):
            
            # torch.cuda.empty_cache()
            inr.train()
            if sub_array_num>1:
                graph_T = graph.time[-1]+1
                time_indexes = divide_array_indexes(graph_T, sub_array_num)
                print('time1:', time()-start)
                start = time()
                
                for tidx in time_indexes:
                    n_samples = len(tidx)
                    graph0 = Data()
                    booltensor = torch.isin(graph.time, tidx)
                    graph0.images = graph.feat[booltensor].to(device)
                    graph0.pos = graph.space_emb[booltensor].to(device)
                    graph0.batch = graph.time[booltensor].to(device)
                    graph0.modulations = graph.latent_vector.to(device)
                    
                    print('time2', time()-start)
                    start = time()
                    
                    outputs = outer_step(
                        inr,
                        graph0,
                        inner_steps,
                        alpha,
                        is_train=True,
                        return_reconstructions=False,
                        gradient_checkpointing=True,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                    )
                    print('time3', time()-start)
                    start = time()
                    
                    print('memory:', torch.cuda.memory_allocated()/1024**3)
                    optimizer.zero_grad()
                    outputs["loss"].backward(create_graph=False)
                    
                    nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
                    print('time4', time()-start)
                    start = time()
                    optimizer.step()
                    loss = outputs["loss"].cpu().detach()
                    fit_train_mse += loss.item() * n_samples / graph_T.item()
                    
                    if use_rel_loss:
                        rel_train_mse += outputs["rel_loss"].item() * n_samples / graph_T.item()
                    print('time5', time()-start)
                    start = time()
                    
            else:
                n_samples = len(graph)

                #modify for outer_step function
                graph.images = graph.feat.to(device)
                graph.pos = graph.space_emb.to(device)
                graph.batch = graph.time.to(device)
                graph.modulations = graph.latent_vector.to(device) #torch.zeros_like(graph.latent_vector)
                # graph.to(device)
                
                outputs = outer_step(
                    inr,
                    graph,
                    inner_steps,
                    alpha,
                    is_train=True,
                    return_reconstructions=False,
                    gradient_checkpointing=False,
                    use_rel_loss=use_rel_loss,
                    loss_type="mse",
                )
                
                optimizer.zero_grad()
                outputs["loss"].backward(create_graph=False)
                nn.utils.clip_grad_value_(inr.parameters(), clip_value=1.0)
                optimizer.step()
                loss = outputs["loss"].cpu().detach()
                fit_train_mse += loss.item() * n_samples
                
                if use_rel_loss:
                    rel_train_mse += outputs["rel_loss"].item() * n_samples
            
        train_loss = fit_train_mse / (ntrain)
        
        
        if use_rel_loss:
            rel_train_loss = rel_train_mse / ntrain
        print('time:', time()-start)
        print('train loss')
        print(train_loss)

        if True in (step_show, step_show_last):
            for substep, graph in enumerate(val_loader):
                inr.eval()
                
                if sub_array_num>1:
                    graph_T = graph.time[-1]+1
                    time_indexes = divide_array_indexes(graph_T, sub_array_num)
                    
                    for tidx in time_indexes:
                        n_samples = len(tidx)
                        graph0 = Data()
                        booltensor = torch.isin(graph.time, tidx)
                        graph0.images = graph.feat[booltensor].to(device)
                        graph0.pos = graph.space_emb[booltensor].to(device)
                        graph0.batch = graph.time[booltensor].to(device)
                        graph0.modulations = graph.latent_vector.to(device)
                        
                        outputs = outer_step(
                            inr, 
                            graph0, 
                            inner_steps, 
                            alpha, 
                            is_train=False, 
                            return_reconstructions=False, 
                            use_rel_loss=use_rel_loss,
                            loss_type="mse",
                        )
                        
                        loss = outputs['loss']
                        fit_test_mse += loss.item() * n_samples/graph_T.item()
                        
                        if use_rel_loss:
                            rel_test_mse += outputs['rel_loss'].item() * n_samples/graph_T.item()
                else:
                    n_samples = len(graph)
                    
                    
                    
                    graph.images = graph.feat.to(device)
                    graph.pos = graph.space_emb.to(device)
                    graph.batch = graph.time.to(device)
                    graph.modulations = graph.latent_vector.to(device) #torch.zeros_like(graph.latent_vector)
                    # graph.modulations = torch.zeros_like(graph.latent_vector)
                    # graph.to(device)

                    outputs = outer_step(
                        inr,
                        graph,
                        inner_steps,
                        alpha,
                        is_train=False,
                        return_reconstructions=False,
                        gradient_checkpointing=False,
                        use_rel_loss=use_rel_loss,
                        loss_type="mse",
                    )

                    loss = outputs["loss"]
                    fit_test_mse += loss.item() * n_samples

                    if use_rel_loss:
                        rel_test_mse += outputs["rel_loss"].item() * n_samples

            test_loss = fit_test_mse / ntest

            if use_rel_loss:
                rel_test_loss = rel_test_mse / ntest
            
            print(f'{step}, test loss')
            print(test_loss)
            # print(rel_test_loss)

            if cfg.wandb.use_wandb:
                wandb.log(
                    {
                        "test_rel_loss": rel_test_loss,
                        "train_rel_loss": rel_train_loss,
                        "test_loss": test_loss,
                        "train_loss": train_loss,
                    },
                    step=step
                )
        
            if test_loss < best_loss:
                best_loss = test_loss

                # try:
                #     savepath = f"{RESULTS_DIR}/{run_name}.pt"
                # except:
                savepath = f'/pscratch/sd/g/gzhao27/INR/SOMA/{current_date_str+run_name}.pt'
                print('savepath:', savepath)
                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "inr": inr.state_dict(),
                        "optimizer_inr": optimizer.state_dict(),
                        "loss": best_loss,
                        "alpha": alpha,
                        "feat_transform": feat_transform,
                        "feat_inv_transform": feat_inv_transform, 
                        # "grid_tr": grid_tr,
                        # "grid_te": grid_te,
                    },
                    savepath,
                )
        
    return rel_test_loss

if __name__ == "__main__":
    main()
    print('finish')