import glob
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from scipy.sparse.linalg import eigsh


def fit_delimiter(string='', length=80, delimiter="="):
    result_len = length - len(string)
    half_len = math.floor(result_len / 2)
    result = delimiter * half_len + string + delimiter * half_len
    return result


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False


def zipdir(path, ziph, include_format):
    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[-1] in include_format:
                filename = os.path.join(root, file)
                arcname = os.path.relpath(os.path.join(root, file), os.path.join(path, '..'))
                ziph.write(filename, arcname)


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def get_normalized_features(X):
    # X.shape=(num_nodes, num_features)
    means = np.mean(X, axis=0)  # mean of features, shape:(num_features,)
    X = X - means.reshape((1, -1))
    stds = np.std(X, axis=0)  # std of features, shape:(num_features,)
    X = X / stds.reshape((1, -1))
    return X, means, stds


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # row sum
    deg_mat_row = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))
    # column sum
    # deg_mat_col = np.asmatrix(np.diag(np.sum(adj_mat, axis=0)))
    deg_mat = deg_mat_row

    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == 'com_lap_mat':
        # Combinatorial
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == 'wid_rw_normd_lap_mat':
        # For ChebConv
        rw_lap_mat = np.matmul(np.linalg.matrix_power(deg_mat, -1), adj_mat)
        rw_normd_lap_mat = id_mat - rw_lap_mat
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which='LM', return_eigenvectors=False)[0]
        wid_rw_normd_lap_mat = 2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        return wid_rw_normd_lap_mat
    elif mat_type == 'hat_rw_normd_lap_mat':
        # For GCNConv
        wid_deg_mat = deg_mat + id_mat
        wid_adj_mat = adj_mat + id_mat
        hat_rw_normd_lap_mat = np.matmul(np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat)
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f'ERROR: {mat_type} is unknown.')


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def masked_mae_loss(input, target, mask_value=-1):

    mask = target == mask_value  # 找到 padding 或无效值的位置
    out = np.abs(input[~mask] - target[~mask])
    loss = out.mean()
    return loss

def masked_rmse_loss(input, target, mask_value=-1):

    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = np.sqrt(out.mean())
    return loss

# def maksed_mse_loss(input, target, mask_value=-1, reduction='mean'):
#     mask = target != mask_value  # 注意：mask=True 表示有效位置
#     loss = (input - target) ** 2
#     loss = loss * mask  # 将无效位置置为0
#
#     if reduction == 'none':
#         return loss
#     elif reduction == 'mean':
#         return loss.sum() / mask.sum().clamp(min=1)  # 防止除0
#     elif reduction == 'sum':
#         return loss.sum()
#     else:
#         raise ValueError(f"Unsupported reduction mode: {reduction}")



def top_k_acc(y_true_seq, y_pred_seq, k):
    hit = 0
    # Convert to binary relevance (nonzero is relevant).
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        top_k_rec = y_pred.argsort()[-k:][::-1]
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:
            hit += 1
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)


def MRR_metric(y_true_seq, y_pred_seq):
    """Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item """
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-len(y_pred):][::-1]
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
    return rlt / len(y_true_seq)


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    top_k_rec = y_pred.argsort()[-k:][::-1]
    idx = np.where(top_k_rec == y_true)[0]
    if len(idx) != 0:
        return 1
    else:
        return 0


def calculate_recall(label_pois, pred_pois, k_values):
    recalls = {k: 0 for k in k_values}  # Initialize recall counters for each k
    total_relevant = len(label_pois)  # Total number of relevant points

    for y_true, y_pred in zip(label_pois, pred_pois):
        for k in k_values:
            top_k_rec = y_pred.argsort()[-k:][::-1]  # Get top k predictions
            if y_true in top_k_rec:
                recalls[k] += 1

    # Calculate Recall@k for each k
    for k in k_values:
        recalls[k] /= total_relevant  # Normalize by total relevant points

    return recalls


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """ next poi metrics """
    # AP: area under PR curve
    # But in next POI rec, the number of positive sample is always 1. Precision is not well defined.
    # Take def of mAP from Personalized Long- and Short-term Preference Learning for Next POI Recommendation
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-k:][::-1]
    r_idx = np.where(rec_list == y_true)[0]
    if len(r_idx) != 0:
        return 1 / (r_idx[0] + 1)
    else:
        return 0

def mAP_metric_full_sequence(y_true_seq, y_pred_seq, k):
    total_mAP = 0.0
    total_relevant = 0

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            total_mAP += 1 / (r_idx[0] + 1)
            total_relevant += 1

    if total_relevant > 0:
        return total_mAP / total_relevant
    else:
        return 0.0

def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """ next poi metrics """
    # Mean Reciprocal Rank: Reciprocal of the rank of the first relevant item
    y_true = y_true_seq[-1]
    y_pred = y_pred_seq[-1]
    rec_list = y_pred.argsort()[-len(y_pred):][::-1]
    r_idx = np.where(rec_list == y_true)[0][0]
    return 1 / (r_idx + 1)


def MRR_metric_full_sequence(y_true_seq, y_pred_seq):
    total_MRR = 0.0
    total_relevant = 0

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[::-1]  # Get sorted predictions
        r_idx = np.where(rec_list == y_true)[0][0]  # Find index of true POI
        total_MRR += 1 / (r_idx + 1)  # Add reciprocal rank
        total_relevant += 1

    # Calculate average MRR
    if total_relevant > 0:
        return total_MRR / total_relevant
    else:
        return 0.0


def calculate_f1_score(y_true_seq, y_pred_seq, k_values):
    f1_scores = {k: 0 for k in k_values}  # Initialize F1-score counters for each k
    total_relevant = len(y_true_seq)  # Total number of relevant points

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        for k in k_values:
            top_k_rec = y_pred.argsort()[-k:][::-1]  # Get top k predictions
            if y_true in top_k_rec:
                precision = 1 / k  # Precision@k
                recall = 1  # Recall@k (since we are considering only one true label)
                f1 = 2 * (precision * recall) / (precision + recall)  # F1-score@k
                f1_scores[k] += f1

    # Calculate average F1-score@k for each k
    for k in k_values:
        f1_scores[k] /= total_relevant  # Normalize by total relevant points

    return f1_scores

def array_round(x, k=4):
    # For a list of float values, keep k decimals of each element
    return list(np.around(np.array(x), k))
def remove_consecutive_duplicates(user_df, col='area_id'):

    # 标记出和前一个值不同的行（即连续段落的开始）
    mask = user_df[col] != user_df[col].shift()
    return user_df[mask].reset_index(drop=True)
import pandas as pd

def remove_consecutive_duplicates_with_avg_time(user_df, col='area_id', time_col='timestamp'):
    # 确保时间列是 datetime 类型
    user_df = user_df.copy()
    user_df[time_col] = pd.to_datetime(user_df[time_col])

    # 创建 segment_id，标记每一段连续不同的 col（area_id）
    segment_id = (user_df[col] != user_df[col].shift()).cumsum()
    user_df['segment_id'] = segment_id

    # 指定需要保留的辅助列
    preserved_cols = [
        'user_id', 'area_tag', 'day_of_week', 'new_trajectory_id',
        'mask', 'area_id_mask', 'time_mask', 'area_tag_mask', 'norm_in_day_time'
    ]
    preserved_cols = [c for c in preserved_cols if c in user_df.columns]

    # 分组聚合
    agg_dict = {col: 'first', time_col: 'mean', 'segment_id': 'count'}
    for c in preserved_cols:
        agg_dict[c] = 'first'

    result = user_df.groupby('segment_id').agg(agg_dict).reset_index(drop=True)

    # 重命名 segment_id 为 duplicates_num
    result = result.rename(columns={'segment_id': 'duplicates_num'})

    return result


