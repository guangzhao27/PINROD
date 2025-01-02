import torch
from torch_geometric.data import Dataset, Data
from torch.utils.data import DataLoader
# from torch_geometric.loader import DataLoader
import argparse

from utils.data.unstructure_dataset import GraphNavierStokes, collate_graph_inr, GraphSomaDataset, GraphBurgers
from coralsoma.load_modulations import graph_ode_inr_predict, load_graph_modulations
import sys
# sys.path.append(str(Path(__file__).parents[1]))
sys.path.append('/pscratch/sd/g/gzhao27/INR/coral')
from coral.utils.models.load_inr import create_inr_instance, load_inr_model

from coral.mlp import Derivative
from coral.utils.models.scheduling import ode_scheduling
from torchdiffeq import odeint

def test_burgers():
    print('Burgers test')
    datapath_dict  = {
        '/pscratch/sd/g/gzhao27/INR/data/1D_Burgers_Sols_Nu0.01.hdf5':([1, 2, 4], 0.01)
    }
    trainset = GraphBurgers(
        datapath_dict=datapath_dict,
        latent_dim=128,
        missing_rate=0.5,
    )
    
    graph = trainset[0]
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=2, shuffle=True, )
    graph = next(iter(train_loader))
    assert graph.cor.size(-1) == 1, 'feature size is not correct'
    assert graph.space_emb.size(-1) == 1
    

def test_soma():
    print('SOMA test')
    device = torch.device('cuda')
    
    trainset = GraphSomaDataset(
        data_path='/global/cfs/cdirs/m4259/ecucuzzella/soma_ppe_data/ml_converted/month_1/thedataset-impliciBottomDrag.hdf5',
        train_num=10, 
        feature_set=[10],
        space_factor=1,
        time_factor=1, 
        latent_dim=256,
    )
    feat_ori = trainset[0].feat
    
    feat_nor, inv_nor = trainset.create_normalize_from_dataset()
    trainset.update_feat_transform(feat_nor)

    graph = trainset[0]
    feat_new = trainset[0].feat
    
    assert (inv_nor(feat_new)-feat_ori).var() < 1e-5, 'the inverse transform is not correct'
    
    
    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=3, shuffle=True, )
    graph = next(iter(train_loader))
    graph = graph.to(device)
    
    ori_feat = inv_nor(graph.feat)
    assert ori_feat.size(-1) == 1
    
    assert graph.cor.size(-1) == 3, 'length of graph is not correct'
    assert graph.feat.size(-1) == 1, 'feature size is not correct'
    
    savepath = f'/pscratch/sd/g/gzhao27/INR/SOMA/test.pt'
    torch.save(
        {
            "feat_transform": feat_nor,
            "feat_inv_transform": inv_nor, 
            # "grid_tr": grid_tr,
            # "grid_te": grid_te,
        },
        savepath,
    )
    

def test_ns():
    print("NS test")
    trainset = GraphNavierStokes(datanum=10, latent_dim=128)

    assert trainset.mask.shape == (10, 64, 64, 50), f"Expected shape (10, 64, 64, 50), but got {trainset.mask.shape}"
    assert len(trainset[0].feat) == len(trainset[0].time), "Expected length to be same, "


    train_loader = DataLoader(dataset=trainset, collate_fn=collate_graph_inr, batch_size=10, shuffle=True, )
    graph = next(iter(train_loader))
    # assert graph.latent_vector[graph.time].size(-1) == 128, "index time for latent vector"

    input_dim = graph.cor.size(-1)
    output_dim = graph.feat.size(-1)
    
    tmp = torch.load('./NS_test.pt')
    latent_dim = tmp["cfg"].inr.latent_dim
    space_factor = tmp["cfg"].data.space_factor
    seed = tmp["cfg"].data.seed
    
    trainset = GraphNavierStokes(split='train', ssub=space_factor, datanum=100, missing_rate=0.5, latent_dim=latent_dim)
    
    inr, alpha = load_inr_model(
        './',
        'NS_test',
        data_to_encode=None, 
        input_dim=input_dim,
        output_dim=output_dim,
    )
    load_graph_modulations(
        trainset,
        inr,
        inner_steps=3,
        alpha=alpha,
        batch_size=4,
    )
    
    timestamps_train = torch.arange(0, 50, 1).float().cuda()
    modulations = torch.rand(100, latent_dim).cuda()
    modulations[:50] = 1
    
    model = Derivative(1, latent_dim, hidden_c=512, depth=3).cuda()
    modulations = modulations.reshape(2, 50, latent_dim)
    modulations = modulations.permute(0, 2, 1)
    z_pred = ode_scheduling(odeint, model, modulations, timestamps_train, epsilon=0)
    assert z_pred.shape == (2, 128, 50)
    
    image_pred, image_pred_loss, z_pred_loss = graph_ode_inr_predict(model, inr, graph)


# inr = ModulatedSiren(
#             dim_in=input_dim,
#             dim_hidden=cfg.inr.hidden_dim,
#             dim_out=output_dim,
#             num_layers=cfg.inr.depth,
#             w0=cfg.inr.w0,
#             w0_initial=cfg.inr.w0,
#             use_bias=True,
#             modulate_scale=cfg.inr.modulate_scale,
#             modulate_shift=cfg.inr.modulate_shift,
#             use_latent=cfg.inr.use_latent,
#             latent_dim=cfg.inr.latent_dim,
#             modulation_net_dim_hidden=cfg.inr.hypernet_width,
#             modulation_net_num_layers=cfg.inr.hypernet_depth,
#             last_activation=cfg.inr.last_activation,
#         ).to(device)
parser = argparse.ArgumentParser()
parser.add_argument('name', type=str, default='NS', help='test name')

args = parser.parse_args()

if "NS" in args.name:
    test_ns()

if "SOMA" in args.name:
    test_soma()
    
if "Burgers" in args.name:
    test_burgers()

print('end')