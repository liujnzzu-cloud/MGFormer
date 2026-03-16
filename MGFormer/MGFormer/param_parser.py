"""Parsing the parameters."""
import argparse

import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # device = torch.device('cpu')
else:
    device = torch.device('cpu')


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GETNext.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed')
    parser.add_argument('--device',
                        type=str,
                        default=device,
                        help='')
    # Data
    parser.add_argument('--data-adj-mtx',
                        type=str,
                        default='dataset/AIS/graph_A.csv',
                        help='Graph adjacent path')
    parser.add_argument('--data-node-feats',
                        type=str,
                        default='dataset/AIS/graph_X.csv',
                        help='Graph node features path')
    parser.add_argument('--data-train',
                        type=str,
                        default='dataset/AIS/train.csv',
                        help='Training data path')
    parser.add_argument('--data-val',
                        type=str,
                        default='dataset/AIS/val.csv',
                        help='Validation data path')
    parser.add_argument('--short-traj-thres',
                        type=int,
                        default=2,
                        help='Remove over-short trajectory')
    parser.add_argument('--time-units',
                        type=int,
                        default=48,
                        help='Time unit is 0.5 hour, 24/0.5=48')

    parser.add_argument('--time-feature',
                        type=str,
                        default='time',
                        help='The name of time feature in the data')
    parser.add_argument('--weekdays-feature',
                        type=str,
                        default='day_of_week',
                        help='The name of weeekday feature in the data')

    # Model hyper-parameters
    parser.add_argument('--poi-embed-dim',
                        type=int,
                        default=64,
                        help='AREA embedding dimensions')
    parser.add_argument('--mmsi-embed-dim',
                        type=int,
                        default=32,
                        help='MMSI embedding dimensions')
    parser.add_argument('--cat-embed-dim',
                        type=int,
                        default=32,
                        help='Cat embedding dimensions')
    parser.add_argument('--length-embed-dim',
                        type=int,
                        default=32,
                        help='Length embedding dimensions')
    parser.add_argument('--time-embed-dim',
                        type=int,
                        default=32,  # 32 64 128 256
                        help='Time embedding dimensions')
    parser.add_argument('--gcn-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for gcn')
    parser.add_argument('--gcn-nhid',
                        type=list,
                        default=[32, 64],
                        help='List of hidden dims for gcn layers')
    parser.add_argument('--transformer-nhid',
                        type=int,
                        default=1024,
                        help='Hid dim in TransformerEncoder')
    parser.add_argument('--transformer-nlayers',
                        type=int,
                        default=2, #1，2，3，4，5
                        help='Num of TransformerEncoderLayer')
    parser.add_argument('--transformer-nhead',
                        type=int,
                        default=2,
                        help='Num of heads in multiheadattention')
    parser.add_argument('--transformer-dropout',
                        type=float,
                        default=0.3,
                        help='Dropout rate for transformer')

    parser.add_argument('--geometry-embed-dim',
                        type=int,
                        default=64,#32 64 128 256
                        help='Geometry embedding dimensions')
    parser.add_argument('--sog-embed-dim',
                        type=int,
                        default=64,
                        help='Sog embedding dimensions')
    parser.add_argument('--cog-embed-dim',
                        type=int,
                        default=64,
                        help='Cog embedding dimensions')
    parser.add_argument('--draught-embed-dim',
                        type=int,
                        default=64,
                        help='Draught embedding dimensions')
    parser.add_argument('--node-attn-nhid',
                        type=int,
                        default=128,
                        help='Node attn map hidden dimensions')
    # parser.add_argument('--time-loss-weight',
    #                     type=int,
    #                     default=5,
    #                     help='Scale factor for the time loss term')

    # normalization mean and std
    parser.add_argument('--lon-mean',
                        type=float,
                        default=121.428651,
                        help='Longitude mean for normalization')

    parser.add_argument('--lon-std',
                        type=float,
                        default=1.143346,
                        help='Longitude std for normalization')

    parser.add_argument('--lat-mean',
                        type=float,
                        default=37.346333,
                        help='Latitude mean for normalization')
    parser.add_argument('--lat-std',
                        type=float,
                        default=0.944584,
                        help='Latitude std for normalization')
    parser.add_argument('--sog-mean',
                        type=float,
                        default=12.965431,
                        help='Sog mean for normalization')
    parser.add_argument('--sog-std',
                        type=float,
                        default=2.965200,
                        help='Sog std for normalization')

    #loss weight

    parser.add_argument('--geometry-loss-weight',
                        type=float,
                        default=5,
                        help='Scale factor for the geometry loss term')

    parser.add_argument('--sog-loss-weight',
                        type=float,
                        default=1,
                        help='Scale factor for the sog loss term')

    parser.add_argument('--cog-loss-weight',
                        type=float,
                        default=1,
                        help='Scale factor for the cog loss term')



    # Training hyper-parameters
    parser.add_argument('--batch',
                        type=int,
                        default=20,
                        help='Batch size.')
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',
                        type=float,
                        default=0.1,
                        help='Learning rate scheduler factor')
    parser.add_argument('--weight_decay',
                        type=float,
                        default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--mask_weight',
                        type=float,
                        default=3.0,
                        help='Weight for masked positions in loss calculation')

    # Experiment config
    parser.add_argument('--save-weights',
                        action='store_true',
                        default=True,
                        help='whether save the model')
    parser.add_argument('--save-embeds',
                        action='store_true',
                        default=False,
                        help='whether save the embeddings')
    parser.add_argument('--workers',
                        type=int,
                        default=0,
                        help='Num of workers for dataloader.')
    parser.add_argument('--project',
                        default='runs/train',
                        help='save to project/name')
    parser.add_argument('--name',
                        default='exp',
                        help='save to project/name')
    parser.add_argument('--exist-ok',
                        action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False, help='Disables CUDA training.')
    parser.add_argument('--mode',
                        type=str,
                        default='client',
                        help='python console use only')
    parser.add_argument('--port',
                        type=int,
                        default=64973,
                        help='python console use only')

    return parser.parse_args()
