import os
import math
import h5py
import torch
import random
import scipy.io
import numpy as np
import xarray as xr
from itertools import product
from einops import rearrange
from argparse import ArgumentParser
from functools import partial

# from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from mmap_ninja import RaggedMmap
from time import time

# def creat_dataset(train_num, val_num, test_num, train_p_list, val_p_list, test_p_list)

def normalize_fn(x, mean, std):
    assert x.size(0) == mean.size(0)
    # mean = mean.reshape(x.shape)
    return (x - mean) / std

def inv_normalize_fn(x, mean, std):
    # mean = mean.reshape(x.shape)
    # assert x.size(0) == mean.size(0)
    x = x.reshape(-1, *mean.shape)
    # print(x.shape)
    # print(mean.shape)
    mean = mean.to(x.device)
    std = std.to(x.device)
    result = x * std + mean
    result = result.reshape(-1, mean.size(-1))
    # print('result:', result.shape)
    return  result

def my_generator(directory):
    """
    Generator function to yield numpy arrays from .npy files in a directory.

    Args:
        directory (str): Path to the directory containing .npy files.

    Yields:
        ndarray: The contents of each .npy file.
    """
    # List and sort the .npy files
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        print(f"Loading: {file_path}")  # Optional logging
        data = np.load(file_path)
        yield data

class TemporalDatasetWithCode(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid, latent_dim=64, dataset_name=None, data_to_encode=None):
        """
        Args:
            dataset
            v (torch.Tensor): Dataset values, with shape (N Dx Dy C T). Where N is the
            number of trajectories, Dx the size of the first spatial dimension, Dy the size
            of the second spatial dimension, C the number of channels (ususally 1), and T the
            number of timestamps.
            grid (torch.Tensor): Coordinates, with shape (N Dx Dy 2). We suppose that we have
            same grid over time.
            latent_dim (int, optional): Latent dimension of the code. Defaults to 64.
        """
        N = v.shape[0]
        T = v.shape[-1]
        self.v = v
        self.c = grid  # repeat_coordinates(grid, N).clone()
        self.output_dim = self.v.shape[-2]
        self.input_dim = self.c.shape[-2]
        self.z = torch.zeros((N, latent_dim, T))
        self.latent_dim = latent_dim
        self.T = T
        self.dataset_name = dataset_name
        self.set_data_to_encode(data_to_encode)


    def set_data_to_encode(self, data_to_encode):
        self.data_to_encode = data_to_encode
        dataset_name = self.dataset_name
        N = self.v.shape[0]
        T = self.v.shape[-1]

        self.index_value = None
        if (data_to_encode is not None) and (dataset_name is not None):
            self.index_value = KEY_TO_INDEX[dataset_name][data_to_encode]
            self.z = torch.zeros((N, self.latent_dim, T))
            self.output_dim = 1

        if data_to_encode is None:
            c = len(KEY_TO_INDEX[dataset_name].keys())
            # one code for the height / vorticity
            # if c == 1, we squeeze it
            self.z = torch.zeros((N, self.latent_dim, c, T)).squeeze(-2)

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        """The tempral dataset returns whole trajectories, identified by the index.

        Args:
            idx (int): idx of the trajectory

        Returns:
            sample_v (torch.Tensor): the trajectory with shape (Dx Dy C T)
            sample_z (torch.Tensor): the codes with shape (L T)
            sample_c (torch.Tensor): the spatial coordinates (Dx Dy 2)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.index_value is not None:
            sample_v = self.v[idx, ..., self.index_value, :]
        else:
            sample_v = self.v[idx, ...]

        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        """How to save efficiently the updated codes.

        Args:
            z_values (torch.Tensor): the updated latent code for the whole trajectory.
            idx (int): idx of the trajectory.
        """
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class GraphSomaDataset(Dataset):
    '''
    data path: "/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5"
    hdf5_file.keys(): foward_0 ... foward_99
    data shape: torch.Size([60, 100, 100, 30, 17])
    '''
    def __init__(self, data_path,
                data_num = 10, 
                train_num=20, 
                feature_set = None, 
                space_factor=1,
                time_factor=1,
                initial_step=10, 
                test_ratio=0.1,
                latent_dim=128, 
                split='train', 
                p_transform=None, 
                feature_transform=None,
                data_save_dir=None, 
                mmap_dir=None,
                sub_array_num=1, 
                ):
        self.name = "SOMA"
        self.data_path =data_path # 
        
        self.inital_step = initial_step
        self.space_factor = space_factor
        self.time_factor = time_factor
        self.feature_set = feature_set
        self.latent_dim = latent_dim
        
        
        self.hdf5_file = h5py.File(self.data_path, 'r')
        self.keys = list(self.hdf5_file.keys()) # keys are 100 forward # data shape is (30, 100)
        self.p_transform = p_transform
        self.feature_transform=feature_transform
        
        if split == 'train':
            self.idx_list = list(range(train_num))
            self.keys = self.keys[:train_num]
        elif split == 'val':
            self.idx_list = list(range(train_num, train_num+data_num))
            self.keys = self.keys[train_num:train_num+data_num]
        else:
            self.idx_list = list(range(-data_num, 0))
            self.keys = self.keys[-data_num:]
            
        if data_save_dir:
            self.data_save_dir = data_save_dir
        else:
            self.data_save_dir = os.path.join(
                '/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_graph_save', 
                f's{self.space_factor}t{self.time_factor}')
            os.makedirs(self.data_save_dir, exist_ok=True)
        
        self.mmap_dir = mmap_dir #/pscratch/sd/g/gzhao27/INR/SOMA/results/soma_mmap_save
        if self.mmap_dir:
            # RaggedMmap.from_generator(out_dir=self.mmap_dir, 
            #                            sample_generator=my_generator(raw_np_dir), 
            #                            batch_size=2)
            self.mmap_data = RaggedMmap(self.mmap_dir)
            
        self.sub_array_num = sub_array_num
        
            
        # this function generate self.T in its function
        self.data_processing() # generate more self properties
        
        if self.sub_array_num > 1:
            base_size = self.T // self.sub_array_num
            extra = self.T % self.sub_array_num
            
            # Array to record the length of each subarray
            self.sub_array_T = [base_size + 1 if i < extra else base_size for i in range(self.sub_array_num)]
        
        # self.dataset = [
        #     self.reduce_resolution(torch.from_numpy(self.hdf5_file[key][:]).permute(1, 2, 3, 0, 4))
        #     for key in self.keys
        # ]
        
        # self.latent_vectors = [
        #     torch.zeros(data.size(3), self.latent_dim)
        #     for key in self.keys
        # ]
        
        self.latent_vectors = []
        for key in self.keys:
            # xx = self.hdf5_file[key][:]
            # reduced_T = len(xx[::time_factor])
            # data = self.reduce_resolution(torch.from_numpy(self.hdf5_file[key][:]).permute(1, 2, 3, 0, 4))
            if self.sub_array_num <= 1:
                self.latent_vectors.append(torch.zeros(self.T, self.latent_dim))
            else:
                for tt in self.sub_array_T:
                    self.latent_vectors.append(torch.zeros(tt, self.latent_dim))
        
    def reduce_resolution(self, _data):
        """
        Apply reduced resolution both in spatial and temporal dimensions.
        """
        # Reduce spatial and temporal dimensions according to reduced_resolution and reduced_resolution_t
        _data = _data[
            ::self.space_factor, 
            ::self.space_factor, 
            ::self.space_factor, 
            ::self.time_factor
        ]
        
        # Apply feature set reduction if provided
        # if self.feature_set is not None:
        #     _data = _data[..., self.feature_set]
        
        return _data
    
    def data_processing(self):
        # _data = self.hdf5_file[self.keys[0]][:] # just take first data to get data shape information
        _data = self.load_raw_data(0)
        # _data = torch.from_numpy(_data)
        _data = _data.permute(1, 2, 3, 0, 4) 
        data = _data[..., :-1]
        sx, sy, sz = data.shape[0:3]
        total_T = _data.size(3)

        _data = _data[::self.space_factor, 
                      ::self.space_factor, 
                      ::self.space_factor, 
                      ::self.time_factor
                      ]

        assert len(self.feature_set) == 1
        _data = _data[..., self.feature_set]

        self.mask = _data[..., 0, 0]> -1000
        self.mu, self.sigma = self.gen_normalize_value(_data) # self.mu and self.sigma is the average over the first single data with index=0, you should not do that!, that is hard to update
        
        
        # generate graph coordinates
        T = _data.size(3)
        self.T = T
        cor = self.mask.nonzero()
        spacial_emb = self.S_embedding(sx, sy, sz, cor)
        time_list = [torch.ones(len(cor), dtype=torch.int)*t for t in range(T)]
        
        self.cor_t = cor.repeat(T, 1)
        self.spacial_emb_t = spacial_emb.repeat(T, 1)
        self.time_t = torch.cat(time_list, dim=0)
        
        # # feature should leave in get function
        # feat_t = _data[self.cor_t[:, 0], self.cor_t[:, 1], self.cor_t[:, 2], self.time_t]

    def S_embedding(self, sx, sy, sz, cor):
        x = torch.arange(sx, dtype=torch.float32)
        y = torch.arange(sy, dtype=torch.float32)
        z = torch.arange(sz, dtype=torch.float32)
        X, Y, Z = torch.meshgrid(x, y, z)
        
        lin_emb = lambda x, y: 2.0*x/(y-1) -1.0
        X = lin_emb(X, sx)
        Y = lin_emb(Y, sy)
        Z = lin_emb(Z, sz)
        
        self.grid = torch.stack((X, Y, Z), dim=-1)
        self.grid = self.grid[::self.space_factor, 
                              ::self.space_factor, 
                              ::self.space_factor, 
                              ]
        
        spatial_embedding = self.grid[cor[:, 0], cor[:, 1], cor[:, 2]]
        
        return spatial_embedding

    def gen_normalize_value(self, data):
        # three dimensional data
        mask_shape = self.mask.shape
        # num_new_dims = len(data.shape) - len(mask_shape) - 1
        expanded_mask = self.mask.view(*mask_shape, 1)
        expanded_mask = expanded_mask.expand(*mask_shape, *data.shape[3:-1])

        mu = torch.zeros(data.shape[-1])
        std = torch.zeros(data.shape[-1])

        for i in range(data.shape[-1]):
            non_zero_data = data[..., i][expanded_mask]
            mu[i] = non_zero_data.mean()
            std[i] = non_zero_data.std()
        
        return mu, std
    
    def create_normalize_from_dataset(self):
        assert self.feature_transform is None, 'feature transform should be none to create new normlaization'
        
        feat_list = []
        p_list = []
        for i in range(len(self.keys)):
            graph = self.getitem(i)
            T = graph.time.max().item()+1
            # p_list.append(graph.ped_para)
            for t in range(T):
                tidx = (graph.time==t)
                feat_list.append(graph.feat[tidx])
            p_list.append(graph.pde_parameter.item())
        feat_tensor = torch.stack(feat_list)
        p_tensor = torch.tensor(p_list)
        
        feat_mean = feat_tensor.mean(dim=0)
        feat_mean = torch.cat([feat_mean]*T)
        feat_std = (feat_tensor - feat_tensor.mean(dim=0)).std()
        
        
        feat_transform = partial(normalize_fn, mean=feat_mean, std=feat_std)
        inv_feat_transform = partial(inv_normalize_fn, mean=feat_mean, std=feat_std)
        
        return feat_transform, inv_feat_transform
        
            
    def update_feat_transform(self, feature_transform):
        self.feature_transform = feature_transform
        
    def update_p_transform(self, p_transform):
        self.p_transform = p_transform

    # def normalize_data(self, data):
    #     # three dimensional data
    #     mask_shape = self.mask.shape
    #     num_new_dims = len(data.shape) - len(mask_shape) - 1
    #     expanded_mask = self.mask.view(*mask_shape, *[1]*num_new_dims)
    #     expanded_mask = expanded_mask.expand(*mask_shape, *data.shape[3:-1])

    #     for i in range(data.shape[-1]):
    #         non_zero_data = data[..., i][expanded_mask]
    #         data[..., i][expanded_mask] = (non_zero_data - self.mu[i])/self.sigma[i]
            
    def __len__(self):
        return len(self.keys)*self.sub_array_num
    
    def load_raw_data(self, idx):
        if self.mmap_dir:
            rawidx = self.idx_list[idx]
            return torch.from_numpy(self.mmap_data[rawidx])
        
        key = self.keys[idx]
        _data = torch.from_numpy(self.hdf5_file[key][:])
        
        return _data
        
    def getitem(self, idx):
        # Y (trajectory) data dimension should be batch*sx*sy*sz*time*D

        _data = self.load_raw_data(idx)
        _data = _data.permute(1, 2, 3, 0, 4)
        data = _data[..., :-1]

        pde_parameter = _data[0, 0, 0,0,  -1:]
        if self.p_transform:
            pde_parameter = self.p_transform(pde_parameter)
            
        
        #reduce datasize
        data = data[
                    ::self.space_factor, 
                    ::self.space_factor, 
                    ::self.space_factor, 
                    ::self.time_factor,
                    ]
        if self.feature_set is not None:
            data = data[..., self.feature_set]
        
        # self.normalize_data(data)
        
        
        feat_t = data[self.cor_t[:, 0], self.cor_t[:, 1], self.cor_t[:, 2], self.time_t]
        # feat_t_ori = feat_t.clone()
        if self.feature_transform:
            feat_t = self.feature_transform(feat_t)

        # change to datapoint and store as a dataset
        graph = Data(
            cor=self.cor_t, time=self.time_t, feat=feat_t, 
            T=torch.tensor(data.size(3)), latent_vector=self.latent_vectors[idx], pde_parameter=pde_parameter,
            space_emb=self.spacial_emb_t, 
            # feat_ori=feat_t_ori,
            )
        return graph
    
    def __getitem__(self, idx):
        graph = self.getitem(idx)
        # if_feat_transform = 'frame_normalize' if self.feature_transform else ''
        # graph_path = os.path.join(self.data_save_dir, 
        #                           self.keys[idx]+if_feat_transform+str(self.latent_dim)+'.pt'
        #                           )
        # if os.path.exists(graph_path):
        #     graph = torch.load(graph_path)
        # else:
        #     graph = self.getitem(idx)
        #     torch.save(graph, graph_path)
        return graph
            
    def __del__(self):
        # Ensure the file is closed when the dataset object is deleted
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
            
    
    # trainset.update_latent_vector(i, z0[latent_idx: latent_idx+tempT])
    def update_latent_vector(self, idx, tensor):
        self.latent_vectors[idx] = tensor



class GraphBurgers(Dataset):
    def __init__(self, 
                 datapath_dict,
                 latent_dim,
                 missing_rate,
                 space_factor=1, time_factor=1,
                 p_transform=None,
                 ):
        # datapath_dict includes datapath and the index_list of corresponding datapath, and parameter value of this datapath
        # dataset include ['t-coordinate', 'tensor', 'x-coordinate']
        # tensor shape (N, tdim, xdim)
        # tdim: 201+1, xdim:1024

        super().__init__()
        
        # self.datapath_dict = datapath_dict
        self.missing_rate = missing_rate
        self.latent_dim = latent_dim
        self.space_factor = space_factor
        self.time_factor = time_factor
        self.p_transform = p_transform
        
        # assert self.space_factor is None
        # assert self.time_factor is None
        
        feature_list = []
        
        dataset = {}
        i = 0
        
        for datapath, (index_list, p) in datapath_dict.items():
            tensor = torch.from_numpy(h5py.File(datapath)['tensor'][index_list, ::time_factor, ::space_factor])
            tc = torch.from_numpy(h5py.File(datapath)['t-coordinate'][::time_factor])
            xc = torch.from_numpy(h5py.File(datapath)['x-coordinate'][::space_factor])
            
            if self.missing_rate>0:
                mask = torch.rand(tensor.shape)>self.missing_rate
            else:
                mask = torch.ones(tensor.shape, dtype=torch.bool)
            
            
            
            for index in range(tensor.size(0)):
                # every index every t, generate cordinates and features
                cor_list = []
                feat_list = []
                time_list = []
                SE_list = []       
                # TE_list = []     
                p_list = []
                
                T = mask.size(1)
                for t in range(T):
                    tmpmask = mask[index, t, :]
                    tmptensor = tensor[index, t, :]
                    cor = tmpmask.nonzero()
                    spacial_embedding = xc[tmpmask].reshape(-1, 1)
                    # time_embedding = torch.ones(len(cor))*tc[t]
                    cor_list.append(cor)
                    feat_list.append(tmptensor[tmpmask])
                    time_list.append(torch.ones(len(cor), dtype=torch.int)*t)
                    SE_list.append(spacial_embedding)
                    # TE_list.append(time_embedding)
                
                cor_t = torch.cat(cor_list, dim=0)
                feat_t = torch.cat(feat_list, dim=0)
                feat_t = feat_t.reshape(-1, 1)
                time_t = torch.cat(time_list, dim=0)
                SE_t = torch.cat(SE_list, dim=0)
                # TE_t = torch.cat(TE_list, dim=0)
                pde_parameter=torch.tensor(p)
                if self.p_transform:
                    pde_parameter = self.p_transform(pde_parameter)
                datapoint = Data(cor=cor_t, time=time_t, 
                                 feat=feat_t, T=T, latent_vector=torch.zeros(T, self.latent_dim), 
                                 space_emb=SE_t, time_emb=tc, pde_parameter=pde_parameter)
                dataset[f"{i}"] = datapoint
                i+=1
                
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        graph = self.dataset[f"{key}"]
        
        return graph
        
        

class GraphNavierStokes(Dataset):
    # properties included in grpah of all time frame:
    # graph.cor: coordinate collections, concatenate all time frame coordinate together
    # graph.time: all time stamples of each coordinate
    # T: time duration of this episode
    # latent_vector: encoded latent vector, initally zero and will be updated
    # space_emb: spatical embedding of graph.cor
    # all other dataset should also includes all these properties to run inr.py and train.py
    
    def __init__(
            self, 
            ssub=1, 
            datapath='/pscratch/sd/g/gzhao27/INR/data/ns_V1e-3_N5000_T50.mat', 
            latent_dim=128, 
            split='train', 
            datanum = 1000,
            trainnum = 1000, 
            missing_rate=0.75,
            ):
        
        super().__init__()

        self.path = datapath
        self.split = split
        self.missing_rate = missing_rate
        self.latent_dim = latent_dim
        self.ssub = ssub
        
        # Dataloading
        datashape = h5py.File(datapath)['u'].shape
        if split == 'train':
            
            data = h5py.File(datapath)['u'][..., ::ssub, ::ssub,  :datanum]
        elif split == 'val':
            data = h5py.File(datapath)['u'][..., ::ssub, ::ssub, trainnum:trainnum+datanum]  #data shape (50, 64, 64, 100)
        elif split == 'test':
            data = h5py.File(datapath)['u'][..., ::ssub, ::ssub, -datanum:]
        tensor = torch.from_numpy(data)
        tensor = tensor.permute(3, 1, 2, 0)

        self.height, self.width = datashape[1], datashape[2]
        self.dataset, self.mask = self.noisy_data_processing(tensor)
        
    def noisy_data_processing(self, tensor):
        if self.missing_rate>0:
            mask = torch.rand(tensor.shape) > self.missing_rate
        else:
            mask = torch.ones(tensor.shape, dtype=torch.bool)
        dataset = {}
        for index in range(len(mask)):
            cor_list = []
            feat_list = []
            time_list = []
            SE_list = []

            datapoint = {}
            T = mask.size(-1)
            for t in range(T):
                tmpmask = mask[index, ..., t]
                tmptensor = tensor[index, ..., t]
                cor =tmpmask.nonzero()
                spacial_embedding = self.S_embedding(self.height, self.width, cor)
                cor_list.append(cor)
                feat_list.append(tmptensor[tmpmask])
                time_list.append(torch.ones(len(cor), dtype=torch.int)*t)
                SE_list.append(spacial_embedding)

            cor_t = torch.cat(cor_list, dim=0)
            feat_t = torch.cat(feat_list, dim=0)
            feat_t = feat_t.reshape(-1, 1)
            time_t = torch.cat(time_list, dim=0)
            spacial_emb_t = torch.cat(SE_list, dim=0)
            datapoint = Data(cor=cor_t, time=time_t, feat=feat_t, T=T, latent_vector=torch.zeros(T, self.latent_dim), space_emb=spacial_emb_t)
            dataset[f"{index}"] = datapoint

        return dataset, mask

    def S_embedding(self, image_height, image_width, cor):
        x_coords, y_coords = torch.meshgrid(
                torch.arange(image_height, dtype=torch.float32),
                torch.arange(image_width, dtype=torch.float32),
                indexing='ij'
            )

        x_coords = 2.0 * x_coords / (image_width - 1) - 1.0
        y_coords = 2.0 * y_coords / (image_height - 1) - 1.0

        spatial_grid = torch.stack([x_coords, y_coords], dim=-1)
        spatial_embedding = spatial_grid[cor[:, 0]*self.ssub, cor[:, 1]*self.ssub]

        return spatial_embedding

    def len(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        graph = self.dataset[f"{key}"]
        
        return graph
    
def collate_graph_inr(data_list):
    data_list_new = []
    batched_data = {}
    
    time_bias = 0 # the bias used to differentiate different batch of time, so time can act as index
    for i, data in enumerate(data_list):
        datatmp = data.clone()
        datatmp.time = data.time + time_bias
        time_bias += data.T
        data_list_new.append(datatmp)
    batched_data = Batch.from_data_list(data_list_new)

    return batched_data


class NavierStokes(Dataset):
    def __init__(
        self, datapath, nx, sub=1, T=20, t_interval=1, n_train=None, n_test=None, missing_rate = 0.75, missing_same = True, train_mask = None):
        self.S = nx // sub
        self.T = T
        self.sub = sub
        self.t_interval = t_interval
        self.n_train = n_train
        self.n_test = n_test
        
        if n_test and missing_same:
            assert train_mask is not None, 'Provide a train mask if missing positions are the same between train, test'
        
        # Missing rate designation
        """
        There are two variants of missing sensors. 
        First, [v1] missing sensors are the same between training and testing
        Second, [v2] missing sensors of training set are different from testing
        Argument: missing_same controls that. [v1] missing_same = True, [v2] missing_same = False
        """
        self.missing_rate = missing_rate
        self.remaining_rate = 1 - missing_rate
        
        
        # Dataloading
        try:
            data = h5py.File(datapath)['u']
        except:
            data = scipy.io.loadmat(datapath)['u']
            data = np.array(data).transpose(3, 1, 2, 0)
        print('Loaded data: {}'.format(data.shape))
        
        if n_train:
            self.a = torch.tensor(data[0 : 1, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
            self.u = torch.tensor(data[1 : self.T + 1, ::sub, ::sub, :n_train], dtype=torch.float).transpose(0, 3)
            
            ## Addressing missing rate            
            self.train_support_mask = self.get_mask()
            
            
        if n_test:
            self.a = torch.tensor(data[0 : 1, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3)      # channel dimension = 1
            self.u = torch.tensor(data[1 : self.T + 1, ::sub, ::sub, -n_test:], dtype=torch.float).transpose(0, 3) # channel dimension = 1
            if missing_same:
                self.test_support_mask = train_mask
            else:
                self.test_support_mask = self.get_mask()
        
        if n_train and n_test:
            raise ValueError
        if not n_train and not n_test:
            raise ValueError
            
        self.get_mesh()
            
    def get_mask(self):
        H, W = self.a.shape[1:3]
        n_support = int(H * W * self.remaining_rate)
        support_mask = torch.zeros(H, W) # get spatial dimension
        loc = list(product(list(range(H)), list(range(W)))) # get all possible combination of H and W
        random.shuffle(loc)
        for h, w in loc[:n_support]:
            support_mask[h,w] = 1
        return support_mask
    
    
    def get_mesh(self):
        # Please use this mesh if need be.
        # geometry locations (x, y)
        mesh1 = torch.tensor(np.linspace(0, 1, self.S), dtype=torch.float)
        mesh2 = torch.tensor(np.linspace(0, 1, self.S), dtype=torch.float)
        mesh1 = mesh1.reshape(self.S, 1, 1).repeat([1, self.S, 1])
        mesh2 = mesh2.reshape(1, self.S, 1).repeat([self.S, 1, 1])
        self.mesh = torch.cat((mesh1, mesh2), dim=-1) # (S x S, 2) 
        
    def __len__(self):
        return self.a.shape[0]

    def __getitem__(self, idx):
        return self.a[idx].unsqueeze(-2), self.u[idx].unsqueeze(-2)

        # return xout, yout
        
def create_burgers_dataset(missing_rate=0.5, space_factor=1, time_factor=1, train_num=100):
    trainnum = train_num
    valnum = 100
    testnum = 100
    train_p_list=(0.001, 0.002, 0.004, 0.02, 0.04, 0.1)
    p_mean = np.array(train_p_list).mean()
    p_std = np.array(train_p_list).std()
    p_transform = lambda tensor: (tensor - p_mean) / p_std
    p_invtransform = lambda tensor: (tensor * p_std) + p_mean
    
    val_p_list = (0.01, )
    test_p_list = (0.002, )
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
    
    trainset = GraphBurgers(
        datapath_dict=traindict,
        latent_dim=128,
        missing_rate=missing_rate,
        space_factor=space_factor,
        time_factor=time_factor,
        p_transform=p_transform,
    )
    valset = GraphBurgers(
        datapath_dict=valdict,
        latent_dim=128,
        missing_rate=0.,
        space_factor=space_factor,
        time_factor=time_factor,
        p_transform=p_transform,
    )
    testset = GraphBurgers(
        datapath_dict=testdict,
        latent_dim=128,
        missing_rate=0.,
        p_transform=p_transform
    )
    return trainset, valset, testset, p_mean, p_std


def soma_claculate_p_normalize(data_path):
    p_list = []
    with h5py.File(data_path, 'r') as f:
        keys = list(f.keys())
        for key in keys:
            _data = f[key][:]
            p_list.append(_data[0, 0, 0, 0, -1])
    
    p_array = np.array(p_list)
    return p_array.mean(), p_array.std()