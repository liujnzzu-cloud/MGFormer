import logging
import logging
import math
import os
import ast
import pathlib
import pickle
import zipfile
import torch.nn.functional as F
from pathlib import Path
from utils import remove_consecutive_duplicates_with_avg_time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model import GCN, NodeAttnMap, MMSIEmbeddings, CategoryEmbeddings, TransformerModel, \
    FuseEmbeddings2, GGNN, Weekday2Vec, LengthEmbeddings, DraughtEmbedding, \
    CoordEmbedding, TimeEmbedding, CogEmbedding, SogEmbedding, FuseEmbeddingsSpatialContext, FuseEmbeddingsStatic, \
    FuseEmbeddingsMotion
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss, calculate_recall, mAP_metric_full_sequence, \
    MRR_metric_full_sequence, calculate_f1_score


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')  # 自动递增保存目录的路径
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)

    # Build poi graph (built from train_df)
    print('Loading poi graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2, args.feature3, args.feature4, args.feature5, args.feature6)
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2},{args.feature3},{args.feature4},{args.feature5}, {args.feature5}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")

    num_pois = raw_X.shape[0]
    num_features = raw_X.shape[1] - 1   # checkin_cnt + 其他数值特征
    X = np.zeros((num_pois, num_features), dtype=np.float32)
    X[:, 0] = raw_X[:, 1].astype(np.float32)
    X[:, 1:] = raw_X[:, 2:].astype(np.float32)
    logging.info(f"After one hot encoding, X.shape: {X.shape}")

    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')  # 标准化的随机游走拉普拉斯矩阵

    # poi id to index
    nodes_df = pd.read_csv(args.data_node_feats)
    poi_ids = list(set(nodes_df['region_sequence'].tolist()))

    if '<m>' in poi_ids:
        poi_ids.remove('<m>')
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))
    poi_id2idx_dict['<m>'] = len(poi_id2idx_dict)

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    # MMSI id to index
    print("缺失的 mmsi 数量：", train_val_df['mmsi'].isna().sum())

    mmsi_ids = [str(each) for each in list(set(train_val_df['mmsi'].to_list()))]
    mmsi_id2idx_dict = dict(zip(mmsi_ids, range(len( mmsi_ids))))

    # Vessel_type id to index
    vessel_type_ids = [str(each) for each in list(set(train_val_df['vessel_type'].to_list()))]
    num_type = len(vessel_type_ids)
    vessel_type_id2idx_dict = dict(zip(vessel_type_ids, range(len(vessel_type_ids))))

    # Length id to index
    length_ids = [str(each) for each in list(set(train_val_df['length'].to_list()))]
    length_id2idx_dict = dict(zip(length_ids, range(len(length_ids))))

    # MMSI idx to type idx
    mmsi_idx2type_idx_dict = {}
    for i, row in train_val_df.iterrows():
        mmsi_idx2type_idx_dict[mmsi_id2idx_dict[str(row['mmsi'])]] = \
            vessel_type_id2idx_dict[row['vessel_type']]

    # MMSI idx to Length idx
    mmsi_idx2length_idx_dict = {}
    for i, row in train_val_df.iterrows():
        mmsi_idx2length_idx_dict[mmsi_id2idx_dict[str(row['mmsi'])]] = \
            length_id2idx_dict[str(row['length'])]

    # Print mmsi-trajectories count
    traj_list = list(set(train_df['trajectory_id'].tolist()))

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.df = train_df
            self.traj_seqs = []  # traj id: mmsi id + traj no.
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                if traj_df.isnull().values.any():
                    continue
                # 恢复值
                geometry_mask = traj_df['geometry_mask'].to_list()
                speed_mask = traj_df['sog_mask'].to_list()
                cog_mask = traj_df['cog_mask'].to_list()
                start_keypoint_mask_ids = traj_df['start_keypoint_mask'].to_list()
                start_keypoint_mask_idx = [
                    poi_id2idx_dict['<m>'] if each == '<m>' else poi_id2idx_dict[int(float(each))]
                    for each in start_keypoint_mask_ids
                ]
                end_keypoint_mask_ids = traj_df['end_keypoint_mask'].to_list()
                end_keypoint_mask_idx = [
                    poi_id2idx_dict['<m>'] if each == '<m>' else poi_id2idx_dict[int(float(each))]
                    for each in end_keypoint_mask_ids
                ]
                time_feature = traj_df[args.time_feature].to_list()
                draught_feature = traj_df['draught'].to_list()


                # 真实值
                geometry = traj_df['geometry'].to_list()
                speed = traj_df['sog'].to_list()
                cog = traj_df['time'].to_list()
                start_keypoint_ids = traj_df['start_keypoint'].to_list()
                start_keypoint_idx = [poi_id2idx_dict[int(float(each))] for each in start_keypoint_ids]
                end_keypoint_ids = traj_df['end_keypoint'].to_list()
                end_keypoint_idx = [poi_id2idx_dict[int(float(each))] for each in end_keypoint_ids]
                time_feature = traj_df[args.time_feature].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(geometry_mask) - 1):
                    input_seq.append((geometry_mask[i], speed_mask[i], cog_mask[i], start_keypoint_mask_idx[i], end_keypoint_mask_idx[i], time_feature[i], draught_feature[i]))
                for i in range(len(geometry) - 1):
                    label_seq.append((geometry[i], speed[i], cog[i], start_keypoint_idx[i], end_keypoint_idx[i],  time_feature[i], draught_feature[i]))
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, df):
            self.df = df
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []

            for traj_id in tqdm(set(df['trajectory_id'].tolist())):
                mmsi_id = traj_id.split('_')[0]

                # Ger pois idx in this trajectory
                traj_df = df[df['trajectory_id'] == traj_id]

                # 恢复值
                geometry_mask = traj_df['geometry_mask'].to_list()
                speed_mask = traj_df['sog_mask'].to_list()
                cog_mask = traj_df['cog_mask'].to_list()
                start_keypoint_mask_ids = traj_df['start_keypoint_mask'].to_list()
                start_keypoint_mask_idx = []
                for each in start_keypoint_mask_ids:
                    if each == "<m>":
                        # poi_mask_idxs.append(0)
                        start_keypoint_mask_idx.append(poi_id2idx_dict['<m>'])
                    elif each in poi_id2idx_dict.keys():
                        start_keypoint_mask_idx.append(poi_id2idx_dict[int(float(each))])  # 仅使用训练集中出现过的 POI  # 仅使用训练集中出现过的 poi
                    else:
                        continue

                end_keypoint_mask_ids = traj_df['end_keypoint_mask'].to_list()
                end_keypoint_mask_idx = []
                for each in end_keypoint_mask_ids:
                    if each == "<m>":
                        end_keypoint_mask_idx.append(poi_id2idx_dict['<m>'])
                    elif each in poi_id2idx_dict.keys():
                        end_keypoint_mask_idx.append(poi_id2idx_dict[int(float(each))])  # 仅使用训练集中出现过的 POI  # 仅使用训练集中出现过的 poi
                    else:
                        continue

                time_feature = traj_df[args.time_feature].to_list()
                draught_feature = traj_df['draught'].to_list()

                geometry = traj_df['geometry'].to_list()
                speed = traj_df['sog'].to_list()
                cog = traj_df['cog'].to_list()
                start_keypoint_ids = traj_df['start_keypoint'].to_list()
                start_keypoint_idx = []

                for each in start_keypoint_ids:
                    if each in poi_id2idx_dict.keys():
                        start_keypoint_idx.append(poi_id2idx_dict[int(float(each))])  # 仅使用训练集中出现过的 poi
                    else:
                        # Ignore poi if not in training set
                        continue

                end_keypoint_ids = traj_df['end_keypoint'].to_list()
                end_keypoint_idx = []
                for each in end_keypoint_ids:
                    if each in poi_id2idx_dict.keys():
                        end_keypoint_idx.append(poi_id2idx_dict[int(float(each))])  # 仅使用训练集中出现过的 poi
                    else:
                        # Ignore poi if not in training set
                        continue

                time_feature = traj_df[args.time_feature].to_list()


                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(geometry_mask) - 1):
                    input_seq.append((geometry[i], speed_mask[i], cog_mask[i], start_keypoint_idx[i],
                                      end_keypoint_idx[i], time_feature[i], draught_feature[i]))
                for i in range(len(geometry) - 1):
                    label_seq.append(
                        (geometry[i], speed[i], cog[i], start_keypoint_idx[i], end_keypoint_idx[i], time_feature[i], draught_feature[i]))


                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: poi embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    args.gcn_nfeat = X.shape[1]
    poi_embed_model = GGNN(ninput=args.gcn_nfeat,
                           nhid=args.gcn_nhid,
                           noutput=args.poi_embed_dim,
                           dropout=args.gcn_dropout)

    # Node Attn Model
    # node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)

    # %% Model2: mmsi embedding model, nn.embedding
    num_mmsis = len(mmsi_id2idx_dict)
    mmsi_embed_model = MMSIEmbeddings(num_mmsis, args.mmsi_embed_dim)

    # %% Model3: length embedding model, nn.embedding
    num_lengths = len(length_id2idx_dict)
    length_embed_model = LengthEmbeddings(num_lengths, args.length_embed_dim)

    # %% Model4: vessel_type embedding model, nn.embedding
    cat_embed_model = CategoryEmbeddings(num_type, args.cat_embed_dim)

    # %% Model5: draught embedding model, nn.embedding
    draught_embed_model = DraughtEmbedding(args.draught_embed_dim)

    # %% Model6: cog embedding model, nn.embedding
    cog_embed_model = CogEmbedding(args.cog_embed_dim)

    # %% Model7: draught embedding model, nn.embedding
    sog_embed_model = SogEmbedding(args.sog_embed_dim)

    # %% Model8: geometry embedding model, nn.embedding
    geometry_embed_model = CoordEmbedding(args.geometry_embed_dim)

    # %% Model9: Time Model
    time_embed_model = TimeEmbedding(args.time_embed_dim)

    # %% Model10: Motion Group Embedding fusion models
    montion_group_embed_fuse_model = FuseEmbeddingsMotion(args.geometry_embed_dim, args.sog_embed_dim, args.cog_embed_dim, args.draught_embed_dim, args.time_embed_dim)

    # %% Model11:Static Group Embedding fusion models
    static_group_embed_fuse_model = FuseEmbeddingsStatic(args.mmsi_embed_dim, args.length_embed_dim, args.cat_embed_dim)

    # %% Model12:Spatial Context Group Embedding fusion models
    spatial_context_group_embed_fuse_model = FuseEmbeddingsSpatialContext(args.poi_embed_dim)

    args.seq_input_embed = (args.geometry_embed_dim + args.sog_embed_dim + args.cog_embed_dim + args.draught_embed_dim + args.time_embed_dim +
                            args.mmsi_embed_dim + args.length_embed_dim + args.cat_embed_dim + 2 * args.poi_embed_dim)


    # %% Model6: Sequence model
    seq_model = TransformerModel(args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)

    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(mmsi_embed_model.parameters()) +
                                  list(length_embed_model.parameters()) +
                                  list(sog_embed_model.parameters()) +
                                  list(cog_embed_model.parameters()) +
                                  list(geometry_embed_model.parameters()) +
                                  list(draught_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(montion_group_embed_fuse_model.parameters()) +
                                  list(static_group_embed_fuse_model.parameters()) +
                                  list(spatial_context_group_embed_fuse_model.parameters())+
                                  list(seq_model.parameters()),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    def criterion_geometry(y_pred, y_true, mask_value=-1):
        mask = (y_true != mask_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)

        diff = (y_pred - y_true) ** 2
        diff = diff[mask]
        loss = diff.mean()
        return loss


    def criterion_sog(y_pred, y_true, mask_value=-1):
        mask = (y_true != mask_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)

        diff = (y_pred - y_true) ** 2
        diff = diff[mask]
        loss = diff.mean()
        return loss

    def criterion_cog(y_pred, y_true, mask_value=-1):
        mask = (y_true != mask_value)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=y_pred.device)

        y_pred_valid = y_pred[mask]
        y_true_valid = y_true[mask]
        y_pred_rad = torch.deg2rad(y_pred_valid)
        y_true_rad = torch.deg2rad(y_true_valid)

        loss = 1 - torch.cos(y_pred_rad - y_true_rad)
        return loss.mean()

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True,
        factor=args.lr_scheduler_factor)

    def input_traj_to_embeddings(sample, poi_embeddings, mask_geometry_embedding, mask_sog_embedding, mask_cog_embedding):
        traj_id = sample[0]
        input_seq_geometry = [each[0] for each in sample[1]]
        input_seq_sog = [each[1] for each in sample[1]]
        input_seq_cog = [each[2] for each in sample[1]]
        input_seq_start_kp = [each[3] for each in sample[1]]
        input_seq_end_kp = [each[4] for each in sample[1]]
        input_seq_time = [each[5] for each in sample[1]]
        input_seq_draught = [each[6] for each in sample[1]]
        mmsi_id = traj_id.split('_')[0]
        # mmsi_idx = mmsi_id2idx_dict[mmsi_id]
        mmsi_idx = mmsi_id2idx_dict.get(f"{mmsi_id}.0")

        input = torch.LongTensor([mmsi_idx]).to(device=args.device)
        mmsi_embedding = mmsi_embed_model(input)
        mmsi_embedding = torch.squeeze(mmsi_embedding)

        # length
        length_idx = mmsi_idx2length_idx_dict[mmsi_idx]
        length_input = torch.tensor([length_idx]).to(device=args.device)
        length_embedding = length_embed_model(length_input)
        length_embedding = torch.squeeze(length_embedding)
        # vessel_type
        type_idx = mmsi_idx2type_idx_dict[mmsi_idx]
        type_input = torch.tensor([type_idx]).to(device=args.device)
        type_embedding = length_embed_model(type_input)
        type_embedding = torch.squeeze(type_embedding)

        input_seq_embed = []
        mask_positions = []

        for idx in range(len(input_seq_geometry)):
            start_kp_embedding = torch.squeeze(poi_embeddings[input_seq_start_kp[idx]].to(device=args.device))
            end_kp_embedding = torch.squeeze(poi_embeddings[input_seq_end_kp[idx]].to(device=args.device))

            # Geometry embedding
            if input_seq_geometry[idx] == "<m>":
                geometry_embedding = mask_geometry_embedding
                geo_mask = True
            else:
                geo_val = ast.literal_eval(input_seq_geometry[idx])
                geo_tensor = torch.tensor([geo_val], dtype=torch.float32, device=args.device)
                geometry_embedding = torch.squeeze(geometry_embed_model(geo_tensor))
                geo_mask = False

            # SOG embedding
            if input_seq_sog[idx] == "<m>":
                sog_embedding = mask_sog_embedding
                sog_mask = True
            else:
                sog_tensor = torch.tensor([[float(input_seq_sog[idx])]], dtype=torch.float32, device=args.device)
                sog_embedding = torch.squeeze(sog_embed_model(sog_tensor))
                sog_mask = False

            # COG embedding
            if input_seq_cog[idx] == "<m>":
                cog_embedding = mask_cog_embedding
                cog_mask = True
            else:
                cog_tensor = torch.tensor([[float(input_seq_cog[idx])]], dtype=torch.float32, device=args.device)
                cog_embedding = torch.squeeze(cog_embed_model(cog_tensor))
                cog_mask = False

            # draught embedding
            draught_tensor = torch.tensor([input_seq_draught[idx]], dtype=torch.float32, device=args.device)
            draught_embedding = torch.squeeze(draught_embed_model(draught_tensor))

            # time embedding
            time_tensor = torch.tensor([input_seq_time[idx]], dtype=torch.float32, device=args.device)
            time_embedding = torch.squeeze(time_embed_model(time_tensor))

            # embedding
            fused_embedding1 = montion_group_embed_fuse_model(geometry_embedding, sog_embedding, cog_embedding,
                                                              draught_embedding, time_embedding)
            fused_embedding2 = static_group_embed_fuse_model(mmsi_embedding, length_embedding, type_embedding)
            fused_embedding3 = spatial_context_group_embed_fuse_model(start_kp_embedding, end_kp_embedding)

            concat_embedding = torch.cat([fused_embedding1, fused_embedding2, fused_embedding3], dim=-1)
            input_seq_embed.append(concat_embedding)

            mask_positions.append(geo_mask or sog_mask or cog_mask)

        input_seq_embed = torch.stack(input_seq_embed, dim=0)
        mask_positions = torch.tensor(mask_positions, dtype=torch.bool, device=args.device)

        return input_seq_embed, mask_positions

    def get_src_key_padding_mask(batch_padded, device):
        batch_padded = torch.as_tensor(batch_padded, device=device)  # 确保是 Tensor
        if batch_padded.dim() == 3:
            non_pad_mask = (batch_padded[:, :, 0] != -1)
        else:
            non_pad_mask = (batch_padded != -1)

        src_key_padding_mask = ~non_pad_mask  # True 表示要 mask

        return src_key_padding_mask.to(torch.bool).transpose(0, 1).to(device)

    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    # node_attn_model = node_attn_model.to(device=args.device)
    mmsi_embed_model = mmsi_embed_model.to(device=args.device)
    length_embed_model = length_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    draught_embed_model = draught_embed_model.to(device=args.device)
    cog_embed_model = cog_embed_model.to(device=args.device)
    sog_embed_model = sog_embed_model.to(device=args.device)
    geometry_embed_model = geometry_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    montion_group_embed_fuse_model = montion_group_embed_fuse_model.to(device=args.device)
    static_group_embed_fuse_model = static_group_embed_fuse_model.to(device=args.device)
    spatial_context_group_embed_fuse_model = spatial_context_group_embed_fuse_model.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    train_epochs_geometry_mae_list = []
    train_epochs_geometry_rmse_list = []
    train_epochs_lon_mae_list = []
    train_epochs_lon_rmse_list = []
    train_epochs_lat_mae_list = []
    train_epochs_lat_rmse_list = []
    train_epochs_sog_mae_list = []
    train_epochs_sog_rmse_list = []
    train_epochs_cog_mae_list = []
    train_epochs_cog_rmse_list = []
    train_epochs_loss_list = []
    train_epochs_geometry_loss_list = []
    train_epochs_sog_loss_list = []
    train_epochs_cog_loss_list = []

    val_epochs_lon_mae_list = []
    val_epochs_lon_rmse_list = []
    val_epochs_lat_mae_list = []
    val_epochs_lat_rmse_list = []
    val_epochs_sog_mae_list = []
    val_epochs_sog_rmse_list = []
    val_epochs_cog_mae_list = []
    val_epochs_cog_rmse_list = []
    val_epochs_loss_list = []
    val_epochs_geometry_loss_list = []
    val_epochs_sog_loss_list = []
    val_epochs_cog_loss_list = []

    # For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        mmsi_embed_model.train()
        length_embed_model.train()
        cat_embed_model.train()
        draught_embed_model.train()
        cog_embed_model.train()
        sog_embed_model.train()
        geometry_embed_model.train()
        time_embed_model.train()
        montion_group_embed_fuse_model.train()
        static_group_embed_fuse_model.train()
        spatial_context_group_embed_fuse_model.train()
        seq_model.train()  # 用于将模型切换到训练模式



        train_batches_geometry_mae_list = []
        train_batches_geometry_rmse_list = []
        train_batches_lon_mae_list = []
        train_batches_lon_rmse_list = []
        train_batches_lat_mae_list = []
        train_batches_lat_rmse_list = []
        train_batches_sog_mae_list = []
        train_batches_sog_rmse_list = []
        train_batches_cog_mae_list = []
        train_batches_cog_rmse_list = []

        train_batches_loss_list = []
        train_batches_geometry_loss_list = []
        train_batches_sog_loss_list = []
        train_batches_cog_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        # Loop batch
        for b_idx, batch in enumerate(
                train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_mask_positions = []
            batch_seq_labels_geometry = []
            batch_seq_labels_sog = []
            batch_seq_labels_cog = []

            poi_embeddings = poi_embed_model(X, A)
            # 添加 mask embedding
            mask_idx = poi_id2idx_dict['<m>']
            embedding_dim = poi_embeddings.shape[1]
            mask_poi_embedding = nn.Parameter(torch.randn(embedding_dim).to(poi_embeddings.device))
            poi_embeddings = torch.cat([poi_embeddings, mask_poi_embedding.unsqueeze(0)], dim=0)

            geo_embedding_dim = geometry_embed_model.mlp[-1].normalized_shape[0]
            mask_geometry_embedding = torch.randn(geo_embedding_dim, device=args.device)
            mask_sog_embedding = torch.randn(geo_embedding_dim, device=args.device)
            mask_cog_embedding = torch.randn(geo_embedding_dim, device=args.device)

            # Convert input seq to embeddings
            for sample in batch:
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq

                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]

                input_seq_geometry = [each[0] for each in sample[1]]
                label_seq_geometry = [each[0] for each in sample[2]]
                input_seq_sog = [each[1] for each in sample[1]]
                label_seq_sog = [each[1] for each in sample[2]]
                input_seq_cog = [each[2] for each in sample[1]]
                label_seq_cog = [each[2] for each in sample[2]]


                input_seq_embed, mask_positions = input_traj_to_embeddings(sample, poi_embeddings, mask_geometry_embedding, mask_sog_embedding, mask_cog_embedding)

                # check_batch(input_seq_embed, "input_seq_embed_after_embedding")

                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_mask_positions.append(mask_positions)

                batch_input_seqs.append(input_seq)
                # batch_seq_labels_geometry.append(torch.LongTensor(label_seq_geometry))
                label_seq_geometry = [ast.literal_eval(x) if isinstance(x, str) else x for x in label_seq_geometry]
                # batch_seq_labels_geometry.append(torch.FloatTensor(label_seq_geometry))
                batch_seq_labels_geometry.append(torch.tensor(label_seq_geometry, dtype=torch.float64))

                batch_seq_labels_sog.append(torch.FloatTensor(label_seq_sog))
                batch_seq_labels_cog.append(torch.LongTensor(label_seq_cog))

            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True,
                                        padding_value=-1)
            batch_mask_positions_padded = pad_sequence(batch_mask_positions, batch_first=True, padding_value=False) #k可在loss或attention中使用

            label_padded_geometry = pad_sequence(batch_seq_labels_geometry, batch_first=True, padding_value=-1)
            label_padded_sog = pad_sequence(batch_seq_labels_sog, batch_first=True, padding_value=-1)
            label_padded_cog = pad_sequence(batch_seq_labels_cog, batch_first=True, padding_value=-1)

            src_key_padding_mask = get_src_key_padding_mask(batch_padded, device=args.device)

            src_key_padding_mask = src_key_padding_mask.to(device=args.device)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_geometry = label_padded_geometry.to(device=args.device, dtype=torch.long)
            y_sog = label_padded_sog.to(device=args.device, dtype=torch.float)
            y_cog = label_padded_cog.to(device=args.device, dtype=torch.long)
            y_pred_geometry, y_pred_sog, y_pred_cog = seq_model(x, src_key_padding_mask=src_key_padding_mask)


            padding_mask = src_key_padding_mask.transpose(0, 1)  # (B, L)
            weights = torch.ones(batch_mask_positions_padded.shape, dtype=torch.float, device=args.device)
            mask_weight = args.mask_weight  # 例如 2.0 或 3.0
            weights[batch_mask_positions_padded] = mask_weight

            weights_flat = weights.view(-1)

            loss_geometry = criterion_geometry(y_pred_geometry, y_geometry)
            loss_geometry_flat = loss_geometry.view(-1)
            loss_geometry = (loss_geometry_flat * weights_flat).mean()

            loss_sog = criterion_sog(torch.squeeze(y_pred_sog), y_sog)
            loss_sog = (loss_sog * weights_flat).mean()

            loss_cog = criterion_cog(y_pred_cog, y_cog)
            loss_cog_flat = loss_cog.view(-1)
            loss_cog = (loss_cog_flat * weights_flat).mean()


            loss = loss_geometry * args.geometry_loss_weight + loss_sog * args.sog_loss_weight  + loss_cog * args.cog_loss_weight

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            geometry_mae, geometry_rmse = 0, 0
            lon_mae, lon_rmse = 0, 0
            lat_mae, lat_rmse = 0, 0
            sog_mae, sog_rmse = 0, 0
            cog_mae, cog_rmse = 0, 0
            batch_label_geometrys = y_geometry.detach().cpu().numpy()
            batch_label_sogs = y_sog.detach().cpu().numpy()
            batch_label_cogs = y_cog.detach().cpu().numpy()

            batch_pred_pgeometrys = y_pred_geometry.detach().cpu().numpy()
            batch_pred_sogs = y_pred_sog.detach().cpu().numpy()
            batch_pred_cogs = y_pred_cog.detach().cpu().numpy()

            n_samples = len(batch_label_geometrys)
            total_valid_geom = 0

            for label_geom, pred_geom, label_sog, pred_sog, label_cog, pred_cog, seq_len in zip(batch_label_geometrys, batch_pred_pgeometrys, batch_label_sogs, batch_pred_sogs, batch_label_cogs, batch_pred_cogs, batch_seq_lens):
                label_geom = label_geom[:seq_len]  # shape: (seq_len, 2)
                pred_geom = pred_geom[:seq_len]  # shape: (seq_len, 2)
                label_sog = label_sog[:seq_len]
                pred_sog = np.squeeze(pred_sog[:seq_len])
                label_cog = label_cog[:seq_len]
                pred_cog = np.squeeze(pred_cog[:seq_len])

                valid_sog = (label_sog != -1)
                valid_cog = (label_cog != -1)
                valid_geom = (label_geom[:, 0] != -1) & (label_geom[:, 1] != -1)

                # --- geometry (欧氏距离) ---
                # if np.any(valid_geom):
                #     diff_geom = np.linalg.norm(pred_geom[valid_geom] - label_geom[valid_geom], axis=1)
                #     diff_geom_nm = diff_geom / 1852  # 转换为海里
                #     geometry_mae += np.mean(np.abs(diff_geom_nm))
                #     geometry_rmse += np.sqrt(np.mean(diff_geom_nm ** 2))
                if np.any(valid_geom):
                    pred_geom_real = np.zeros_like(pred_geom)
                    label_geom_real = np.zeros_like(label_geom)

                    pred_geom_real[:, 0] = pred_geom[:, 0] * args.lon_std + args.lon_mean
                    pred_geom_real[:, 1] = pred_geom[:, 1] * args.lat_std + args.lat_mean
                    label_geom_real[:, 0] = label_geom[:, 0] * args.lon_std + args.lon_mean
                    label_geom_real[:, 1] = label_geom[:, 1] * args.lat_std + args.lat_mean

                    diff = pred_geom_real[valid_geom] - label_geom_real[valid_geom]

                    geometry_mae += np.mean(np.abs(diff))
                    geometry_rmse += np.sqrt(np.mean(diff ** 2))


                    lon_diff = diff[:, 0]
                    lat_diff = diff[:, 1]

                    lon_mae = np.mean(np.abs(lon_diff))
                    lon_rmse = np.sqrt(np.mean(lon_diff ** 2))
                    lat_mae = np.mean(np.abs(lat_diff))
                    lat_rmse = np.sqrt(np.mean(lat_diff ** 2))


                if np.any(valid_sog):
                    pred_sog_real = pred_sog[valid_sog] * args.sog_std + args.sog_mean
                    label_sog_real = label_sog[valid_sog] * args.sog_std + args.sog_mean

                    sog_mae += np.abs(label_sog_real - pred_sog_real).mean()
                    sog_rmse += np.sqrt(((label_sog_real - pred_sog_real) ** 2).mean())

                if np.any(valid_cog):
                    diff_cog = np.abs(label_cog[valid_cog] - pred_cog[valid_cog])
                    diff_cog = np.minimum(diff_cog, 360 - diff_cog)
                    cog_mae += diff_cog.mean()
                    cog_rmse += np.sqrt((diff_cog ** 2).mean())

                if n_samples > 0:
                    geometry_mae /= n_samples
                    geometry_rmse /= n_samples
                    lon_mae /= n_samples
                    lon_rmse /= n_samples
                    lat_mae /= n_samples
                    lat_rmse /= n_samples
                    sog_mae /= n_samples
                    sog_rmse /= n_samples
                    cog_mae /= n_samples
                    cog_rmse /= n_samples


            train_batches_geometry_mae_list.append(geometry_mae)
            train_batches_geometry_rmse_list.append(geometry_rmse)
            train_batches_lon_mae_list.append(lon_mae)
            train_batches_lon_rmse_list.append(lon_rmse)
            train_batches_lat_mae_list.append(lat_mae)
            train_batches_lat_rmse_list.append(lat_rmse)
            train_batches_sog_mae_list.append(sog_mae)
            train_batches_sog_rmse_list.append(sog_rmse)
            train_batches_cog_mae_list.append(cog_mae)
            train_batches_cog_rmse_list.append(cog_rmse)

            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_geometry_loss_list.append(loss_geometry.detach().cpu().numpy())
            train_batches_sog_loss_list.append(loss_sog.detach().cpu().numpy())
            train_batches_cog_loss_list.append(loss_cog.detach().cpu().numpy())  # 记录损失值

            if (b_idx % (args.batch * 5)) == 0:
                logging.info(
                    f"Epoch:{epoch}, batch:{b_idx}\n"
                    f"Total Loss: {loss.item():.4f}\n"
                    f"Geometry Loss: {loss_geometry.item():.4f}, SOG Loss: {loss_sog.item():.4f}, COG Loss: {loss_cog.item():.4f}\n"
                    f"Mean Geometry MAE: {np.mean(train_batches_geometry_mae_list):.4f} m, "
                    f"RMSE: {np.mean(train_batches_geometry_rmse_list):.4f} m\n"
                      f"Mean Lon MAE: {np.mean(train_batches_lon_mae_list):.4f} m, "
                    f"RMSE: {np.mean(train_batches_lon_rmse_list):.4f} m\n"
                      f"Mean Lat MAE: {np.mean(train_batches_lat_mae_list):.4f} m, "
                    f"RMSE: {np.mean(train_batches_lat_rmse_list):.4f} m\n"
                    f"Mean SOG MAE: {np.mean(train_batches_sog_mae_list):.4f}, "
                    f"RMSE: {np.mean(train_batches_sog_rmse_list):.4f}\n"
                    f"Mean COG MAE: {np.mean(train_batches_cog_mae_list):.4f}°, "
                    f"RMSE: {np.mean(train_batches_cog_rmse_list):.4f}°\n"
                    f"traj_id: {batch[0][0]}\n"
                    f"input_seq_len: {batch_seq_lens[0]}\n"
                    f"{'=' * 100}"
                )

    #     # train end --------------------------------------------------------------------------------------------------------


        poi_embed_model.eval()
        mmsi_embed_model.eval()
        length_embed_model.eval()
        cat_embed_model.eval()
        draught_embed_model.eval()
        cog_embed_model.eval()
        sog_embed_model.eval()
        geometry_embed_model.eval()
        time_embed_model.eval()
        montion_group_embed_fuse_model.eval()
        static_group_embed_fuse_model.eval()
        spatial_context_group_embed_fuse_model.eval()
        seq_model.eval()

        val_batches_lon_mae_list = []
        val_batches_lon_rmse_list = []
        val_batches_lat_mae_list = []
        val_batches_lat_rmse_list = []
        val_batches_sog_mae_list = []
        val_batches_sog_rmse_list = []
        val_batches_cog_mae_list = []
        val_batches_cog_rmse_list = []

        val_batches_loss_list = []
        val_batches_geometry_loss_list = []
        val_batches_sog_loss_list = []
        val_batches_cog_loss_list = []


        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_mask_positions = []
            batch_seq_labels_geometry = []
            batch_seq_labels_sog = []
            batch_seq_labels_cog = []

            poi_embeddings = poi_embed_model(X, A)  # 计算poi的嵌入
            # 添加 mask embedding
            mask_idx = poi_id2idx_dict['<m>']
            embedding_dim = poi_embeddings.shape[1]
            mask_embedding = nn.Parameter(torch.randn(embedding_dim).to(poi_embeddings.device))
            poi_embeddings = torch.cat([poi_embeddings, mask_embedding.unsqueeze(0)], dim=0)

            geo_embedding_dim = geometry_embed_model.mlp[-1].normalized_shape[0]  # 取 geometry_embed_model 输出维度
            mask_geometry_embedding = nn.Parameter(torch.zeros(geo_embedding_dim, device=args.device))
            mask_sog_embedding = nn.Parameter(torch.zeros(geo_embedding_dim, device=args.device))
            mask_cog_embedding = nn.Parameter(torch.zeros(geo_embedding_dim, device=args.device))

            # Convert input seq to embeddings
            for sample in batch:
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]

                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]

                input_seq_geometry = [each[0] for each in sample[1]]
                label_seq_geometry = [each[0] for each in sample[2]]
                input_seq_sog = [each[1] for each in sample[1]]
                label_seq_sog = [each[1] for each in sample[2]]
                input_seq_cog = [each[2] for each in sample[1]]
                label_seq_cog = [each[2] for each in sample[2]]


                input_seq_embed, mask_positions = input_traj_to_embeddings(sample, poi_embeddings,
                                                                           mask_geometry_embedding, mask_sog_embedding,
                                                                           mask_cog_embedding)


                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_mask_positions.append(mask_positions)

                batch_input_seqs.append(input_seq)

                label_seq_geometry = [ast.literal_eval(x) if isinstance(x, str) else x for x in label_seq_geometry]
                batch_seq_labels_geometry.append(torch.tensor(label_seq_geometry, dtype=torch.float64))
                batch_seq_labels_sog.append(torch.FloatTensor(label_seq_sog))
                batch_seq_labels_cog.append(torch.LongTensor(label_seq_cog))

            # Pad seqs for batch training
            batch_padded = pad_sequence(batch_seq_embeds, batch_first=True,
                                        padding_value=-1)  # 形状：(batch_size, max_seq_len, embed_dim)
            batch_mask_positions_padded = pad_sequence(batch_mask_positions, batch_first=True,
                                                       padding_value=False)  # k可在loss或attention中使用
            label_padded_geometry = pad_sequence(batch_seq_labels_geometry, batch_first=True, padding_value=-1)
            label_padded_sog = pad_sequence(batch_seq_labels_sog, batch_first=True, padding_value=-1)
            label_padded_cog = pad_sequence(batch_seq_labels_cog, batch_first=True, padding_value=-1)

            #掩码
            src_key_padding_mask = get_src_key_padding_mask(batch_padded, device=args.device)
            src_key_padding_mask = src_key_padding_mask.to(device=args.device)

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_geometry = label_padded_geometry.to(device=args.device, dtype=torch.long)
            y_sog = label_padded_sog.to(device=args.device, dtype=torch.float)
            y_cog = label_padded_cog.to(device=args.device, dtype=torch.long)
            y_pred_geometry, y_pred_sog, y_pred_cog = seq_model(x, src_key_padding_mask=src_key_padding_mask)

            print(y_pred_sog.min().item(), y_pred_sog.max().item())
            print(y_pred_cog.min().item(), y_pred_cog.max().item())

            def check_nan_inf(name, tensor):
                if torch.isnan(tensor).any():
                    print(f" {name} contains NaN")
                if torch.isinf(tensor).any():
                    print(f" {name} contains Inf")

            check_nan_inf("y_pred_sog", y_pred_sog)
            check_nan_inf("y_sog", y_sog)
            check_nan_inf("y_pred_cog", y_pred_cog)
            check_nan_inf("y_cog", y_cog)

            if torch.isnan(y_pred_sog).any() or torch.isnan(y_pred_cog).any():
                print(f"NaN detected at batch {b_idx}")
                print(f"Input range: {x.min().item():.3e} ~ {x.max().item():.3e}")
                break

            padding_mask = src_key_padding_mask.transpose(0, 1)  # (B, L)
            weights = torch.ones(batch_mask_positions_padded.shape, dtype=torch.float, device=args.device)
            mask_weight = args.mask_weight  # 例如 2.0 或 3.0
            weights[batch_mask_positions_padded] = mask_weight

            weights_flat = weights.view(-1)

            loss_geometry = criterion_geometry(y_pred_geometry, y_geometry)
            loss_geometry_flat = loss_geometry.view(-1)
            loss_geometry = (loss_geometry_flat * weights_flat).mean()

            loss_sog = criterion_sog(torch.squeeze(y_pred_sog), y_sog)
            loss_sog = (loss_sog * weights_flat).mean()

            loss_cog = criterion_cog(y_pred_cog, y_cog)
            loss_cog_flat = loss_cog.view(-1)
            loss_cog = (loss_cog_flat * weights_flat).mean()

            print(f"loss_geometry: {loss_geometry.item():.6f}")
            print(f"loss_sog: {loss_sog.item():.6f}")
            print(f"loss_cog: {loss_cog.item():.6f}")

            loss = loss_geometry * args.geometry_loss_weight + loss_sog * args.sog_loss_weight + loss_cog * args.cog_loss_weight

            geometry_mae, geometry_rmse = 0, 0

            sog_mae, sog_rmse = 0, 0
            cog_mae, cog_rmse = 0, 0
            batch_label_geometrys = y_geometry.detach().cpu().numpy()
            batch_label_sogs = y_sog.detach().cpu().numpy()
            batch_label_cogs = y_cog.detach().cpu().numpy()

            batch_pred_pgeometrys = y_pred_geometry.detach().cpu().numpy()
            batch_pred_sogs = y_pred_sog.detach().cpu().numpy()
            batch_pred_cogs = y_pred_cog.detach().cpu().numpy()  # 将 PyTorch Tensor 转换为 NumPy 数组

            n_samples = len(batch_label_geometrys)
            total_valid_geom = 0
            lon_mae = 0
            lon_rmse = 0
            lat_mae = 0
            lat_rmse = 0

            for label_geom, pred_geom, label_sog, pred_sog, label_cog, pred_cog, seq_len in zip(batch_label_geometrys,
                                                                                                batch_pred_pgeometrys,
                                                                                                batch_label_sogs,
                                                                                                batch_pred_sogs,
                                                                                                batch_label_cogs,
                                                                                                batch_pred_cogs,
                                                                                                batch_seq_lens):

                label_geom = label_geom[:seq_len]  # shape: (seq_len, 2)
                pred_geom = pred_geom[:seq_len]  # shape: (seq_len, 2)
                label_sog = label_sog[:seq_len]
                pred_sog = np.squeeze(pred_sog[:seq_len])
                label_cog = label_cog[:seq_len]
                pred_cog = np.squeeze(pred_cog[:seq_len])

                valid_sog = (label_sog != -1)
                valid_cog = (label_cog != -1)
                valid_geom = (label_geom[:, 0] != -1) & (label_geom[:, 1] != -1)

                if np.any(valid_geom):

                    pred_geom_real = np.zeros_like(pred_geom)
                    label_geom_real = np.zeros_like(label_geom)

                    pred_geom_real[:, 0] = pred_geom[:, 0] * args.lon_std + args.lon_mean  # 经度
                    pred_geom_real[:, 1] = pred_geom[:, 1] * args.lat_std + args.lat_mean  # 纬度
                    label_geom_real[:, 0] = label_geom[:, 0] * args.lon_std + args.lon_mean
                    label_geom_real[:, 1] = label_geom[:, 1] * args.lat_std + args.lat_mean

                    diff = pred_geom_real[valid_geom] - label_geom_real[valid_geom]

                    # 分别计算 lon / lat 的 MAE 和 RMSE
                    lon_diff = diff[:, 0]
                    lat_diff = diff[:, 1]

                    lon_mae = np.mean(np.abs(lon_diff))
                    lon_rmse = np.sqrt(np.mean(lon_diff ** 2))

                    lat_mae = np.mean(np.abs(lat_diff))
                    lat_rmse = np.sqrt(np.mean(lat_diff ** 2))

                # --- sog ---
                if np.any(valid_sog):
                    pred_sog_real = pred_sog[valid_sog] * args.sog_std + args.sog_mean
                    label_sog_real = label_sog[valid_sog] * args.sog_std + args.sog_mean

                    sog_mae += np.abs(label_sog_real - pred_sog_real).mean()
                    sog_rmse += np.sqrt(((label_sog_real - pred_sog_real) ** 2).mean())

                if np.any(valid_cog):
                    diff_cog = np.abs(label_cog[valid_cog] - pred_cog[valid_cog])
                    diff_cog = np.minimum(diff_cog, 360 - diff_cog)
                    cog_mae += diff_cog.mean()
                    cog_rmse += np.sqrt((diff_cog ** 2).mean())

                if n_samples > 0:
                    # geometry_mae /= n_samples
                    lon_mae /= n_samples
                    lon_rmse /= n_samples
                    lat_mae /=n_samples
                    lat_rmse /= n_samples
                    geometry_rmse /= n_samples
                    sog_mae /= n_samples
                    sog_rmse /= n_samples
                    cog_mae /= n_samples
                    cog_rmse /= n_samples


            val_batches_lon_mae_list.append(lon_mae)
            val_batches_lon_rmse_list.append(lon_rmse)
            val_batches_lat_mae_list.append(lat_mae)
            val_batches_lat_rmse_list.append(lat_rmse)
            val_batches_sog_mae_list.append(sog_mae)
            val_batches_sog_rmse_list.append(sog_rmse)
            val_batches_cog_mae_list.append(cog_mae)
            val_batches_cog_rmse_list.append(cog_rmse)

            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_geometry_loss_list.append(loss_geometry.detach().cpu().numpy())
            val_batches_sog_loss_list.append(loss_sog.detach().cpu().numpy())
            val_batches_cog_loss_list.append(loss_cog.detach().cpu().numpy())

            if (b_idx % (args.batch * 5)) == 0:
                logging.info(
                    f"Epoch:{epoch}, batch:{b_idx}\n"
                    f"Val Total Loss: {loss.item():.4f}\n"
                    f"Val Geometry Loss: {loss_geometry.item():.4f}, Val SOG Loss: {loss_sog.item():.4f}, Val COG Loss: {loss_cog.item():.4f}\n"
                    f"Val Mean LON MAE: {np.mean(val_batches_lon_mae_list):.4f} m, "
                    f"RMSE: {np.mean(val_batches_lon_rmse_list):.4f} m\n"
                    f"Val Mean LON MAE: {np.mean(val_batches_lat_mae_list):.4f} m, "
                    f"RMSE: {np.mean(val_batches_lat_rmse_list):.4f} m\n"
                    f"Val Mean SOG MAE: {np.mean(val_batches_sog_mae_list):.4f}, "
                    f"RMSE: {np.mean(val_batches_sog_rmse_list):.4f}\n"
                    f"Val Mean COG MAE: {np.mean(val_batches_cog_mae_list):.4f}°, "
                    f"RMSE: {np.mean(val_batches_cog_rmse_list):.4f}°\n"
                    f"traj_id: {batch[0][0]}\n"
                    f"input_seq_len: {batch_seq_lens[0]}\n"
                    f"{'=' * 100}"
                )
    #     # valid end --------------------------------------------------------------------------------------------------------
    #
    #     # Calculate epoch metrics
    #
        epoch_train_geometry_mae = np.mean(train_batches_geometry_mae_list)
        epoch_train_geometry_rmse = np.mean(train_batches_geometry_rmse_list)
        epoch_train_lon_mae = np.mean(train_batches_lon_mae_list)
        epoch_train_lon_rmse = np.mean(train_batches_lon_rmse_list)
        epoch_train_lat_mae = np.mean(train_batches_lat_mae_list)
        epoch_train_lat_rmse = np.mean(train_batches_lat_rmse_list)
        epoch_train_sog_mae = np.mean(train_batches_sog_mae_list)
        epoch_train_sog_rmse = np.mean(train_batches_sog_rmse_list)
        epoch_train_cog_mae = np.mean(train_batches_cog_mae_list)
        epoch_train_cog_rmse = np.mean(train_batches_cog_rmse_list)


        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_geometry_loss = np.mean(train_batches_geometry_loss_list)
        epoch_train_sog_loss = np.mean(train_batches_sog_loss_list)
        epoch_train_cog_loss = np.mean(train_batches_cog_loss_list)

        epoch_val_lon_mae = np.mean(val_batches_lon_mae_list)
        epoch_val_lon_rmse = np.mean(val_batches_lon_rmse_list)
        epoch_val_lat_mae = np.mean(val_batches_lat_mae_list)
        epoch_val_lat_rmse = np.mean(val_batches_lat_rmse_list)
        epoch_val_sog_mae = np.mean(val_batches_sog_mae_list)
        epoch_val_sog_rmse = np.mean(val_batches_sog_rmse_list)
        epoch_val_cog_mae = np.mean(val_batches_cog_mae_list)
        epoch_val_cog_rmse = np.mean(val_batches_cog_rmse_list)

        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_geometry_loss = np.mean(val_batches_geometry_loss_list)
        epoch_val_sog_loss = np.mean(val_batches_sog_loss_list)
        epoch_val_cog_loss = np.mean(val_batches_cog_loss_list)

    #     # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_geometry_loss_list.append(epoch_train_geometry_loss)
        train_epochs_sog_loss_list.append(epoch_train_sog_loss)
        train_epochs_cog_loss_list.append(epoch_train_cog_loss)
        train_epochs_geometry_mae_list.append(epoch_train_geometry_mae)
        train_epochs_geometry_rmse_list.append(epoch_train_geometry_rmse)
        train_epochs_lon_mae_list.append(epoch_train_lon_mae)
        train_epochs_lon_rmse_list.append(epoch_train_lon_rmse)
        train_epochs_lat_mae_list.append(epoch_train_lat_mae)
        train_epochs_lat_rmse_list.append(epoch_train_lat_rmse)
        train_epochs_sog_mae_list.append(epoch_train_sog_mae)
        train_epochs_sog_rmse_list.append(epoch_train_sog_rmse)
        train_epochs_cog_mae_list.append(epoch_train_cog_mae)
        train_epochs_cog_rmse_list.append(epoch_train_cog_rmse)

        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_geometry_loss_list.append(epoch_val_geometry_loss)
        val_epochs_sog_loss_list.append(epoch_val_sog_loss)
        val_epochs_cog_loss_list.append(epoch_val_cog_loss)
        val_epochs_lon_mae_list.append(epoch_val_lon_mae)
        val_epochs_lon_rmse_list.append(epoch_val_lon_rmse)
        val_epochs_lat_mae_list.append(epoch_val_lat_mae)
        val_epochs_lat_rmse_list.append(epoch_val_lat_rmse)
        val_epochs_sog_mae_list.append(epoch_val_sog_mae)
        val_epochs_sog_rmse_list.append(epoch_val_sog_rmse)

        val_epochs_cog_mae_list.append(epoch_val_cog_mae)
        val_epochs_cog_rmse_list.append(epoch_val_cog_rmse)


        # Monitor loss and score
        monitor_loss = epoch_val_loss


        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"epoch_train_geometry_loss:{epoch_train_geometry_loss:.4f}, "
                     f"epoch_train_sog_loss:{epoch_train_sog_loss:.4f}, "
                     f"epoch_train_cog_loss:{epoch_train_cog_loss:.4f}, "
                     f"epoch_train_geometry_mae:{epoch_train_geometry_mae:.4f}, "
                     f"epoch_train_geometry_rmse:{epoch_train_geometry_rmse:.4f}, "
                      f"epoch_train_lon_mae:{epoch_train_lon_mae:.4f}, "
                     f"epoch_train_lon_rmse:{epoch_train_lon_rmse:.4f}, "
                      f"epoch_train_lat_mae:{epoch_train_lat_mae:.4f}, "
                     f"epoch_train_lat_rmse:{epoch_train_lat_rmse:.4f}, "
                     f"epoch_train_sog_mae:{epoch_train_sog_mae:.4f}, "
                     f"epoch_train_sog_rmse:{epoch_train_sog_rmse:.4f}, "
                     f"epoch_train_cog_mae:{epoch_train_cog_mae:.4f}, "
                     f"epoch_train_cog_rmse:{epoch_train_cog_rmse:.4f}, "
               

              
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"epoch_val_geometry_loss: {epoch_val_geometry_loss:.4f}, "
                     f"epoch_val_sog_loss: {epoch_val_sog_loss:.4f}, "
                     f"epoch_val_cog_loss: {epoch_val_cog_loss:.4f}, "
                     f"epoch_val_lon_mae:{epoch_val_lon_mae:.4f}, "
                     f"epoch_val_lon_rmse:{epoch_val_lon_rmse:.4f}, "
                     f"epoch_val_lat_mae:{epoch_val_lat_mae:.4f}, "
                     f"epoch_val_lat_rmse:{epoch_val_lat_rmse:.4f}, "
                     f"epoch_val_sog_mae:{epoch_val_sog_mae:.4f}, "
                     f"epoch_val_sog_rmse:{epoch_val_sog_rmse:.4f}, "
                     f"epoch_val_cog_mae:{epoch_val_cog_mae:.4f}, "
                     f"epoch_val_cog_rmse:{epoch_val_cog_rmse:.4f}, "
                     )

    #     # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_geometry_loss_list={[float(f"{each:.4f}") for each in train_epochs_geometry_loss_list]}', file=f)
            print(f'train_epochs_sog_loss_list={[float(f"{each:.4f}") for each in train_epochs_sog_loss_list]}',
                  file=f)
            print(f'train_epochs_cog_loss_list={[float(f"{each:.4f}") for each in train_epochs_cog_loss_list]}', file=f)
            print(f'train_epochs_geometry_mae_list={[float(f"{each:.4f}") for each in train_epochs_geometry_mae_list]}', file=f)
            print(f'train_epochs_geometry_rmse_list={[float(f"{each:.4f}") for each in train_epochs_geometry_rmse_list]}', file=f)
            print(f'train_epochs_lon_mae_list={[float(f"{each:.4f}") for each in train_epochs_lon_mae_list]}',
                  file=f)
            print(
                f'train_epochs_lon_rmse_list={[float(f"{each:.4f}") for each in train_epochs_lon_rmse_list]}',
                file=f)
            print(f'train_epochs_lat_mae_list={[float(f"{each:.4f}") for each in train_epochs_lat_mae_list]}',
                  file=f)
            print(
                f'train_epochs_lat_rmse_list={[float(f"{each:.4f}") for each in train_epochs_lat_rmse_list]}',
                file=f)
            print(f'train_epochs_sog_mae_list={[float(f"{each:.4f}") for each in train_epochs_sog_mae_list]}',
                  file=f)
            print(f'train_epochs_sog_rmse_list={[float(f"{each:.4f}") for each in train_epochs_sog_rmse_list]}',
                  file=f)
            print(f'train_epochs_cog_mae_list={[float(f"{each:.4f}") for each in train_epochs_cog_mae_list]}', file=f)
            print(f'train_epochs_cog_rmse_list={[float(f"{each:.4f}") for each in train_epochs_cog_rmse_list]}', file=f)

        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_geometry_loss_list={[float(f"{each:.4f}") for each in val_epochs_geometry_loss_list]}', file=f)
            print(f'val_epochs_sog_loss_list={[float(f"{each:.4f}") for each in val_epochs_sog_loss_list]}', file=f)
            print(f'val_epochs_cog_loss_list={[float(f"{each:.4f}") for each in val_epochs_cog_loss_list]}', file=f)
            print(f'val_epochs_lon_mae_list={[float(f"{each:.4f}") for each in val_epochs_lon_mae_list]}', file=f)
            print(f'val_epochs_lon_rmse_list={[float(f"{each:.4f}") for each in val_epochs_lon_rmse_list]}', file=f)
            print(f'val_epochs_lat_mae_list={[float(f"{each:.4f}") for each in val_epochs_lat_mae_list]}', file=f)
            print(f'val_epochs_lat_rmse_list={[float(f"{each:.4f}") for each in val_epochs_lat_rmse_list]}', file=f)
            print(f'val_epochs_sog_mae_list={[float(f"{each:.4f}") for each in val_epochs_sog_mae_list]}', file=f)
            print(f'val_epochs_sog_rmse_list={[float(f"{each:.4f}") for each in val_epochs_sog_rmse_list]}', file=f)
            print(f'val_epochs_cog_mae_list={[float(f"{each:.4f}") for each in val_epochs_cog_mae_list]}', file=f)
            print(f'val_epochs_cog_rmse_list={[float(f"{each:.4f}") for each in val_epochs_cog_rmse_list]}', file=f)

        logging.info(f"Finall resrult\n"
                     f"mean_epoch_val_lon_mae:{np.mean(val_epochs_lon_mae_list):.4f}, "
                     f"mean_epoch_val_lon_rmse:{np.mean(val_epochs_lon_rmse_list):.4f}, "
                     f"mean_epoch_val_lat_mae:{np.mean(val_epochs_lat_mae_list):.4f}, "
                     f"mean_epoch_val_lat_rmse:{np.mean(val_epochs_lat_rmse_list):.4f}, "
                     f"mean_epoch_val_sog_mae:{np.mean(val_epochs_sog_mae_list):.4f}, "
                     f"mean_epoch_val_sog_rmse:{np.mean(val_epochs_sog_rmse_list):.4f}, "
                     f"mean_epoch_val_cog_mae:{np.mean(val_epochs_cog_mae_list):.4f}, "
                     f"mean_epoch_val_cog_rmse:{np.mean(val_epochs_cog_rmse_list):.4f}, "
                     )



if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    # args.feature1 = 'node_name/poi_id'
    # args.feature2 = 'checkin_cnt'
    # args.feature3 = 'poi_tag'

    args.feature1 = 'checkin_cnt'
    args.feature2 = 'Tanker_ratio'
    args.feature3 = 'Cargo_ratio'
    args.feature4 = 'area_m2'
    args.feature5 = 'latitude'
    args.feature6 = 'longitude'
    train(args)
