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
from torch_geometric.data import DataLoader as GDataLoader
from datetime import datetime
from functools import partial

os.environ["WANDB_DIR"] = '/pscratch/sd/g/gzhao27/INR/coral/wandb'
os.environ["RESULTS_DIR"] = ''

sys.path.append(str(Path(__file__).parents[1]))
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
from coral.utils.plot import show
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.load_inr import create_inr_instance, load_inr_model
from coral.utils.models.get_inr_reconstructions import get_reconstructions
from load_modulations import load_graph_modulations, graph_ode_inr_predict, load_soma_graph_modulations_each_frame
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import (KEY_TO_INDEX, TemporalDatasetWithCode)
from coral.mlp import Derivative, ParameterizedDerivative
from torchdiffeq import odeint
from omegaconf import DictConfig, OmegaConf
from utils.data.unstructure_dataset import (
    GraphNavierStokes, collate_graph_inr, GraphSomaDataset, 
    create_burgers_dataset, soma_claculate_p_normalize,
    create_soma_dataset,
                                            )
from coralsoma.train import train_step, val_step
import optuna

class DetailedMSE():
    def __init__(self, keys, dataset_name="shallow-water-dino", mode="train", n_trajectories=256):
        self.keys = keys
        self.mode = mode
        self.dataset_name = dataset_name
        self.n_trajectories = n_trajectories
        self.reset_dic()

    def reset_dic(self):
        dic = {}
        for key in self.keys:
            dic[f"{key}_{self.mode}_mse"] = 0
        self.dic = dic

    def aggregate(self, u_pred, u_true):
        n_samples = u_pred.shape[0]
        for key in self.keys:
            idx = KEY_TO_INDEX[self.dataset_name][key]
            self.dic[f"{key}_{self.mode}_mse"] += (
                (u_pred[..., idx, :] - u_true[..., idx, :])**2).mean()*n_samples

    def get_dic(self):
        dic = self.dic
        for key in self.keys:
            dic[f"{key}_{self.mode}_mse"] /= self.n_trajectories
        return self.dic  
    
def modulation_fix(modulations, n_samples, T, latent_dim, z_transform=None, device=None):
    if device is not None:
        modulations = modulations.to(device)
    modulations = modulations.reshape(n_samples, T, latent_dim)
    modulations = modulations.permute(0, 2, 1)
    if z_transform is not None:
        modulations = z_transform(modulations)
    return modulations

def ode_pred_z(model, graph, modulations, timestamps, epsilon, parameterized):
    if not parameterized:
        _f = model
    else:
        _f = partial(model, graph.pde_parameter)
    
    z_pred = ode_scheduling(odeint, _f, modulations, timestamps, epsilon)
    
    return z_pred

@hydra.main(config_path="../config/", config_name="ode.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # neceassary for some reason now
    torch.set_default_dtype(torch.float32)
    current_date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # data
    data_dir = cfg.data.dir
    data_path=cfg.data.data_path
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    mmap_dir = cfg.data.mmap_dir
    data_to_encode = cfg.data.data_to_encode
    space_factor = cfg.data.space_factor
    time_factor = cfg.data.time_factor
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len
    missing_rate = cfg.data.missing_rate
    val_missing_rate = cfg.data.val_missing_rate
    sub_array_num = cfg.data.sub_array_num
    
    train_num = cfg.train_num
    p_dim = cfg.p_dim

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay
    gamma_step = cfg.optim.gamma_step
    epochs = cfg.optim.epochs

    # inr
    inr_save_name = cfg.inr.save_name
    inr_save_dir = cfg.inr.save_dir

    inner_steps = cfg.inr.inner_steps

    # dynamics
    model_type = cfg.dynamics.model_type
    hidden = cfg.dynamics.width
    depth = cfg.dynamics.depth
    epsilon = cfg.dynamics.teacher_forcing_init
    epsilon_t = cfg.dynamics.teacher_forcing_decay
    epsilon_freq = cfg.dynamics.teacher_forcing_update
    para_type = cfg.para_type
    assert para_type in ['no-parameter', 'concatenate', 'concatenate_test', 'modulation']
    # assert para_type in ['no-parameter', 'concatenate', 'modulation']
    if para_type == 'no-parameter':
        parameterized = False
    else:
        parameterized = True
    p_mean = 0.0
    p_std = 1.0

    # wandb
    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name
    print('run name:', run_name)
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"), f"wandb/{cfg.wandb.dir}")
        if cfg.wandb.dir is not None
        else None
    )
    
    ode_save_name = cfg.ode_save_name
    epoch_load = -1

    print("run dir given", run_dir)
    
    if cfg.wandb.use_wandb:
        run = wandb.init(
            project=project,
            name=run_name,
            id=run_id,
            dir=None,
        )

        if run_dir is not None:
            os.symlink(run.dir.split("/files")[0], run_dir)

        wandb.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )
        run_name = wandb.run.name

        print("id", run.id)
        print("dir", run.dir)

    root_dir = Path(os.getenv("WANDB_DIR")) / dataset_name

    if data_to_encode is not None:
        model_dir = (
            Path(os.getenv("WANDB_DIR")) /
            dataset_name / data_to_encode / "model"
        )
    else:
        model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "model"

    os.makedirs(str(model_dir), exist_ok=True)

    # we need the latent dim and the space_factor used for training
    if inr_save_name is not None:
        multichannel = False
        tmp = torch.load(os.path.join(inr_save_dir, inr_save_name+'.pt'))
        latent_dim = tmp["cfg"].inr.latent_dim
        # print('fix sub_tr to space_factor')
        # space_factor = tmp["cfg"].data.space_factor
        seed = tmp["cfg"].data.seed
    else:
        raise NotImplementedError("save file is empty")

    set_seed(seed)
    if dataset_name == "NS":
        input_dim=2
        output_dim=1
        trainset = GraphNavierStokes(split='train', ssub=space_factor, datanum=100, missing_rate=missing_rate, latent_dim=latent_dim)
        valset = GraphNavierStokes(split='val', ssub=space_factor, trainnum=100, datanum=100, missing_rate=missing_rate, latent_dim=latent_dim)
        testset = GraphNavierStokes(split='test', datanum=200, missing_rate=missing_rate, latent_dim=latent_dim)
    elif dataset_name == "SOMA":
        input_dim = 3
        output_dim = 1
        feature_set = [10]
        trainset, valset, testset, feat_transform,  feat_inv_transform = create_soma_dataset(
            ntrain, mmap_dir, space_factor, time_factor, 
            latent_dim, missing_rate, val_missing_rate, 
            feature_set, data_path=data_path)
        # data_path = '/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5'
        # # p_mean, p_std = soma_claculate_p_normalize(data_path)
        # print('for soma thedataset-impliciBottomDrag p_mean and p_std pre-calculated: 0.005342233, 0.002689823')
        # p_mean, p_std = 0.005342233, 0.002689823 # for 
        # p_transform = lambda tensor: (tensor - p_mean) / p_std
        # p_invtransform = lambda tensor: (tensor * p_std) + p_mean
        # trainset = GraphSomaDataset(
        #     data_path=data_path,
        #     train_num=80, 
        #     feature_set=[10],
        #     space_factor=4,
        #     time_factor=2, 
        #     latent_dim=latent_dim,
        #     p_transform=p_transform,
        # )
        # valset = GraphSomaDataset(
        #     data_path=data_path,
        #     train_num=10, 
        #     feature_set=[10],
        #     space_factor=4,
        #     time_factor=2, 
        #     latent_dim=latent_dim,
        #     split='val', 
        #     p_transform=p_transform,
        # )
        # testset = GraphSomaDataset(
        #     data_path=data_path,
        #     train_num=10, 
        #     feature_set=[10],
        #     space_factor=4,
        #     time_factor=2, 
        #     latent_dim=latent_dim,
        #     split='test', 
        #     p_transform=p_transform,
        # )
    elif dataset_name == "Burgers":
        input_dim = 1
        output_dim = 1
        trainset, valset, testset, p_mean, p_std = create_burgers_dataset(missing_rate=missing_rate, space_factor=space_factor, time_factor=time_factor, train_num=train_num)
    else:
        raise NotImplementedError(f"The dataset ${dataset_name} does not have a corresponding class.")

    ntrain = len(trainset)
    nval = len(valset)
    ntest = len(testset)

    # # sequence length 
    T_train = trainset[0].T
    T_val = valset[0].T

    dt = 1
    timestamps_train = torch.arange(0, T_train, dt).float().cuda()
    timestamps_val = torch.arange(0, T_val, dt).float().cuda()

    # # trainset coords of shape (N, Dx, Dy, input_dim, T)
    # input_dim = grid_tr.shape[-2]
    # # trainset images of shape (N, Dx, Dy, output_dim, T)
    # output_dim = u_train.shape[-2]

    if inr_save_name is not None:
        inr, alpha = load_inr_model(
            inr_save_dir,
            inr_save_name,
            data_to_encode,
            input_dim=input_dim,
            output_dim=output_dim,
        )
        
        if dataset_name == 'SOMA':
            z_mean, z_std, trainset = load_soma_graph_modulations_each_frame(
                inr_save_name=inr_save_name,
                inr=inr,
                trainset=trainset,
                type='train',
                device=device,
                inner_steps=inner_steps,
                alpha=alpha,
            )
            
            
            z_transform = lambda tensor: (tensor - z_mean) / z_std
            z_invtransform = lambda tensor: (tensor * z_std) + z_mean
            
            # train_p_list=(0.001, 0.002, 0.004, 0.02, 0.04, 0.1)
            # p_mean = np.array(train_p_list).mean()
            # p_std = np.array(train_p_list).std()
            # p_transform = lambda tensor: (tensor - p_mean) / p_std
            # p_invtransform = lambda tensor: (tensor * p_std) + p_mean
            
            _, _, valset = load_soma_graph_modulations_each_frame(
                inr_save_name=inr_save_name,
                inr=inr,
                trainset=valset,
                type='val',
                device=device,
                inner_steps=inner_steps,
                alpha=alpha,
            )
        else:
        #this function requires to change dataset
            raise NotImplementedError

    else:
        raise NotImplementedError

    # create torch dataset
    train_loader = GDataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_graph_inr,
        pin_memory=True,
    )
    val_loader = GDataLoader(
        valset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_graph_inr,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_inr,  #should use collate_graph_ode instead TODO
        num_workers=1,
    )

    c = output_dim
    
    def train_and_evaluate(cfg, trial):
        
        current_date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
        # data
        data_dir = cfg.data.dir
        dataset_name = cfg.data.dataset_name
        ntrain = cfg.data.ntrain
        ntest = cfg.data.ntest
        mmap_dir = cfg.data.mmap_dir
        data_to_encode = cfg.data.data_to_encode
        space_factor = cfg.data.space_factor
        time_factor = cfg.data.time_factor
        seed = cfg.data.seed
        same_grid = cfg.data.same_grid
        seq_inter_len = cfg.data.seq_inter_len
        seq_extra_len = cfg.data.seq_extra_len
        missing_rate = cfg.data.missing_rate
        val_missing_rate = cfg.data.val_missing_rate
        sub_array_num = cfg.data.sub_array_num
        
        train_num = cfg.train_num
        p_dim = cfg.p_dim

        # optim
        batch_size = cfg.optim.batch_size
        batch_size_val = (
            batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
        )
        lr = cfg.optim.lr
        weight_decay = cfg.optim.weight_decay
        gamma_step = cfg.optim.gamma_step
        epochs = cfg.optim.epochs

        # inr
        inr_save_name = cfg.inr.save_name
        inr_save_dir = cfg.inr.save_dir

        inner_steps = cfg.inr.inner_steps

        # dynamics
        model_type = cfg.dynamics.model_type
        hidden = cfg.dynamics.width
        depth = cfg.dynamics.depth
        epsilon = cfg.dynamics.teacher_forcing_init
        epsilon_t = cfg.dynamics.teacher_forcing_decay
        epsilon_freq = cfg.dynamics.teacher_forcing_update
        para_type = cfg.para_type
        assert para_type in ['no-parameter', 'concatenate', 'concatenate_test', 'modulation']

        if para_type == 'no-parameter':
            parameterized = False
        else:
            parameterized = True
        p_mean = 0.0
        p_std = 1.0

        # wandb
        entity = cfg.wandb.entity
        project = cfg.wandb.project
        run_id = cfg.wandb.id
        run_name = cfg.wandb.name
        if not parameterized:
            model = Derivative(c, latent_dim, hidden, depth).cuda()
        else:
            model = ParameterizedDerivative(c, latent_dim, hidden, depth=depth, para_type=para_type, p_dim=p_dim).cuda()
        
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=gamma_step,
            patience=250,
            threshold=0.01,
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-5,
            eps=1e-08,
            verbose=True,
        )
        
        if ode_save_name:
            save_dir = '/pscratch/sd/g/gzhao27/INR/SOMA/results/'
            ode_save_path = os.path.join(inr_save_dir, ode_save_name)
            ode_results = torch.load(ode_save_path)

            cfg = ode_results['cfg']
            ode_results['epoch']
            model_state_dict = ode_results['ode']
            scheduler.load_state_dict(model_state_dict)
            epoch_load = ode_results['epoch']
            optimizer_state_dict = ode_results['ode_optimizer']
            optimizer.load_state_dict(optimizer_state_dict)
            print('load epoch:', epoch_load)
            
            

        best_loss = np.inf

        if multichannel:
            detailed_train_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                            dataset_name,
                                            mode="train",
                                            n_trajectories=ntrain)
            detailed_train_eval_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                            dataset_name,
                                            mode="train_extra",
                                            n_trajectories=ntrain)
            detailed_test_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                            dataset_name,
                                            mode="test",
                                            n_trajectories=ntest)
        else:
            detailed_train_mse = None
            detailed_train_eval_mse = None
            detailed_test_mse = None
            
        for step in range(epochs):
            step_show = step % 20 == 0
            step_show_last = step == epochs - 1

            if step % epsilon_freq == 0:
                epsilon_t = epsilon_t * epsilon
            
            code_train_mse = train_step(
                step, train_loader, model, 
                device, T_train, latent_dim, 
                z_transform, timestamps_train, 
                epsilon_t, parameterized, 
                optimizer, ntrain)
            
            if True in (step_show, step_show_last):
                
                code_val_mse = val_step(
                    step, val_loader, model, 
                    device, T_val, latent_dim,
                    z_transform, timestamps_val, 
                    parameterized, nval)
                
            
                if code_val_mse < best_loss:
                    best_loss = code_val_mse
                    # if code_val_mse < best_loss:
                    #     best_loss = code_val_mse
                    savepath = os.path.join('/pscratch/sd/g/gzhao27/INR/SOMA/', inr_save_name+f'{run_name}-{trial.number}_ode.pt')
                    
                    torch.save(
                        {
                            "cfg":cfg, 
                            "epoch": step,
                            "ode": model.state_dict(),
                            "ode_optimizer": optimizer.state_dict(),
                            "loss": best_loss,
                            "z_mean": z_mean,
                            "z_std": z_std,
                            "epsilon": epsilon_t,
                            "p_mean": p_mean,
                            "p_std": p_std,
                        },
                        savepath,
                    )
            
            scheduler.step(code_train_mse)
            
            trial.report(best_loss, step)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_loss
    
    def objective(trial, cfg: DictConfig):
        
        print('define the hyperparameter')
        cfg.dynamics.width = trial.suggest_int('width', 200, 1000, log=True)
        cfg.dynamics.depth = trial.suggest_int('depth', 3, 9)
        cfg.p_dim = trial.suggest_int('p_dim', 16, 256, log=True)
        cfg.optim.lr = trial.suggest_float('lr', 1e-7, 1e-4, log=True)
        cfg.para_type = trial.suggest_categorical(
            'category', 
            ['concatenate', 'concatenate_test', 'modulation', 'no-parameter']
            )
        # concatenate_test, concatenate on each layer of the whole mlp forward
        print(trial.number, cfg.para_type, cfg.dynamics.depth, cfg.dynamics.width, cfg.p_dim, cfg.optim.lr)
        # cfg.optim.epochs=20
        accuracy = train_and_evaluate(cfg, trial)
        
        return accuracy
    
    storage_url = os.path.join('sqlite:////pscratch/sd/g/gzhao27/INR/SOMA/results', 
                               f"{inr_save_name}{run_name}-ode-finetune.db"
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