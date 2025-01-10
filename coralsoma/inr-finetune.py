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
from utils.data.unstructure_dataset import (
    GraphNavierStokes, 
    collate_graph_inr, 
    GraphSomaDataset, 
    GraphBurgers, 
    create_burgers_dataset, 
    create_soma_dataset,
    )


from coral.losses import batch_mse_rel_fn
from coral.metalearning import graph_outer_step as outer_step
from coral.mlp import MLP, Derivative, ResNet
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, rearrange
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.models.load_inr import create_inr_instance
from datetime import datetime
from time import time
from torch_geometric.data import Data
import optuna
from functools import partial
import pickle
from coralsoma.inr import train_step, validation_step

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
    current_date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
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
    data_path = cfg.data.data_path
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
    val_missing_rate = cfg.data.val_missing_rate
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
        
        trainset, valset, testset, feat_transform,  feat_inv_transform = create_soma_dataset(
            ntrain, mmap_dir, space_factor, time_factor, 
            latent_dim, missing_rate, val_missing_rate, 
            feature_set, data_path, )
        
        # trainset0 = GraphSomaDataset(
        #     data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
        #     train_num=ntrain, 
        #     feature_set=[10],
        #     space_factor=space_factor,
        #     time_factor=time_factor, 
        #     latent_dim=latent_dim,
        #     mmap_dir=mmap_dir,
        #     missing_rate=0.0,
        # )
        # # create a separate create normlize transform function, avoid it to be related to the dataset sampling strategy
        # feat_transform, feat_inv_transform = trainset0.create_normalize_from_dataset() 
        
        # trainset = GraphSomaDataset(
        #     data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
        #     train_num=ntrain, 
        #     feature_set=[10],
        #     space_factor=space_factor,
        #     time_factor=time_factor, 
        #     latent_dim=latent_dim,
        #     mmap_dir=mmap_dir,
        #     missing_rate=missing_rate,
        # )
        # trainset.update_feat_transform(feat_transform)
        # valset = GraphSomaDataset(
        #     data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
        #     train_num=ntrain, 
        #     data_num=10,
        #     feature_set=[10],
        #     space_factor=space_factor,
        #     time_factor=time_factor, 
        #     latent_dim=latent_dim,
        #     split='val',
        #     feature_transform=feat_transform,
        #     mmap_dir=mmap_dir,
        #     missing_rate=val_missing_rate,
        # )
        # testset = GraphSomaDataset(
        #     data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
        #     train_num=ntrain, 
        #     data_num=10,
        #     feature_set=[10],
        #     space_factor=space_factor,
        #     time_factor=time_factor, 
        #     latent_dim=latent_dim,
        #     split='test',
        #     feature_transform=feat_transform,
        #     mmap_dir=mmap_dir,
        # )
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

    def train_and_evaluate(cfg, trial):
        
        
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
        
        set_seed(seed)
        
        # feature transform
        feat_transform, feat_inv_transform = None, None
        
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
        
        
        ntrain = len(trainset)
        nval = len(valset)
        ntest = len(testset)

        train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_inr)
        val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_inr)
        test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, collate_fn=collate_graph_inr)

        print("train", len(trainset))
        print("val", len(valset))

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
            use_rel_loss = step % 10 == 0
            step_show = step % 10 == 0
            step_show_last = step == epochs - 1
            
            train_loss, rel_train_loss = train_step(
                step, train_loader, inr, 
                sub_array_num, device, missing_rate, 
                inner_steps, alpha, 
                use_rel_loss, optimizer, ntrain
                )

            if True in (step_show, step_show_last):
                test_loss, rel_test_loss = validation_step(
                    step, val_loader, inr, 
                    sub_array_num, device, 
                    inner_steps, alpha, use_rel_loss, ntest
                    )
                
                if step<=98:
                    best_loss = test_loss
                else:
                    if test_loss < best_loss:
                        best_loss = test_loss
                        savepath = f'/pscratch/sd/g/gzhao27/INR/SOMA/{trial.number}-{run_name}-finetune.pt'
                        # print('savepath:', savepath)
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
                trial.report(best_loss, step)
                
                if best_loss > 1.4:
                    print('manual prune')
                    raise optuna.exceptions.TrialPruned()
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                
                # if step>98 and test_loss < best_loss:
                #     best_loss = test_loss
        
        return best_loss
        
    
    def objective(trial, cfg: DictConfig):
    
        cfg.inr.w0 = trial.suggest_int('w0', 1, 50, log=True)
        cfg.optim.lr_inr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
        cfg.inr.depth = trial.suggest_int('inr-depth', 6, 12)
        cfg.inr.hidden_dim = trial.suggest_int('inr-hidden-dim', 32, 256, log=True)
        print(trial.number, cfg.inr.w0, cfg.optim.lr_inr, cfg.inr.depth, cfg.inr.hidden_dim )
        # cfg.optim.epochs=20
        accuracy = train_and_evaluate(cfg, trial)
        
        return accuracy
    
    storage_url = os.path.join('sqlite:////pscratch/sd/g/gzhao27/INR/SOMA/results', 
                               f"{run_name}-optuna.db"
                               )
    print(storage_url)
    # # run_name = 'SO-fine-bs2_sf4_tf2_san1_epoch601'
    # storage_url = 'sqlite:///'+f'/pscratch/sd/g/gzhao27/INR/SOMA/results/{run_name}-optuna.db'
    
    study = optuna.create_study(
                                study_name=run_name,
                                direction='minimize', 
                                sampler=optuna.samplers.TPESampler(), 
                                pruner=optuna.pruners.HyperbandPruner(),
                                storage=storage_url,
                                load_if_exists=True,
                                )
    
    #test code part
    # objective(study.trials[19], cfg)
    
    
    study.optimize(partial(objective, cfg=cfg), n_trials=10)
    
    
    return

if __name__ == "__main__":
    main()
    print('finish')