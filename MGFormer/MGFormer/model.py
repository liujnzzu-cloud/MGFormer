import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class NodeAttnMap(nn.Module):
    def __init__(self, in_features, nhid, use_mask=False):
        super(NodeAttnMap, self).__init__()
        self.use_mask = use_mask
        self.out_features = nhid
        self.W = nn.Parameter(torch.empty(size=(in_features, nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * nhid, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, X, A):
        Wh = torch.mm(X, self.W)

        e = self._prepare_attentional_mechanism_input(Wh)

        if self.use_mask:
            e = torch.where(A > 0, e, torch.zeros_like(e))  # mask

        A = A + 1  # shift from 0-1 to 1-2
        e = e * A

        return e

    def _prepare_attentional_mechanism_input(self, Wh):
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GatedGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, num_steps=3):
        super(GatedGraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_steps = num_steps  # GGNN 消息传播步数

        # 线性变换矩阵，用于初始消息传递
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # 候选状态的权重矩阵
        self.candidate_weight = Parameter(torch.FloatTensor(out_features, out_features))
        # 更新门和重置门的参数
        self.update_gate = nn.Linear(out_features * 2, out_features)
        self.reset_gate = nn.Linear(out_features * 2, out_features)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.candidate_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        nn.init.xavier_uniform_(self.update_gate.weight)
        nn.init.xavier_uniform_(self.reset_gate.weight)
        nn.init.zeros_(self.update_gate.bias)
        nn.init.zeros_(self.reset_gate.bias)

    def forward(self, input, adj):
        # input: (num_nodes, in_features)
        # adj: 邻接矩阵 (num_nodes, num_nodes)
        h = torch.mm(input, self.weight)  # 初始节点表示: (num_nodes, out_features)
        if self.bias is not None:
            h = h + self.bias

        # 多步消息传递
        for _ in range(self.num_steps):
            # 消息传递：聚合邻居信息
            m = torch.spmm(adj, h)  # (num_nodes, out_features)

            # 准备门控输入
            gate_input = torch.cat([h, m], dim=1)  # (num_nodes, out_features * 2)

            # 计算更新门和重置门
            z = torch.sigmoid(self.update_gate(gate_input))  # 更新门: (num_nodes, out_features)
            r = torch.sigmoid(self.reset_gate(gate_input))  # 重置门: (num_nodes, out_features)

            # 计算候选状态
            candidate = torch.tanh(torch.mm(r * h, self.candidate_weight))  # (num_nodes, out_features)

            # 更新节点状态
            h = z * h + (1 - z) * candidate

        return h

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout):
        super(GCN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)     #定义 GCN 组件。使用 Leaky ReLU 作为非线性激活函数

        channels = [ninput] + nhid + [noutput]  #channels 构造 GCN 层的输入输出维度：
        for i in range(len(channels) - 1):
            gcn_layer = GraphConvolution(channels[i], channels[i + 1])  #GraphConvolution(channels[i], channels[i + 1]) 创建 GCN 层
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x
class GGNN(nn.Module):
    def __init__(self, ninput, nhid, noutput, dropout, num_steps=3):
        super(GGNN, self).__init__()

        self.gcn = nn.ModuleList()
        self.dropout = dropout
        self.leaky_relu = nn.LeakyReLU(0.2)

        channels = [ninput] + nhid + [noutput]
        for i in range(len(channels) - 1):
            gcn_layer = GatedGraphConvolution(channels[i], channels[i + 1], num_steps=num_steps)
            self.gcn.append(gcn_layer)

    def forward(self, x, adj):
        for i in range(len(self.gcn) - 1):
            x = self.leaky_relu(self.gcn[i](x, adj))

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn[-1](x, adj)

        return x

class MMSIEmbeddings(nn.Module):
    def __init__(self, num_mmsis, embedding_dim):
        super(MMSIEmbeddings, self).__init__()

        self.mmsi_embedding = nn.Embedding(
            num_embeddings=num_mmsis,
            embedding_dim=embedding_dim,
        )  #num_embeddings=num_users：有多少个用户，就创建多少个嵌入向量。embedding_dim=embedding_dim：每个用户的嵌入向量的维度

    def forward(self, mmsi_idx):
        embed = self.mmsi_embedding(mmsi_idx)
        return embed


class LengthEmbeddings(nn.Module):
    def __init__(self, num_lengths, embedding_dim):
        super(LengthEmbeddings, self).__init__()

        self.length_embedding = nn.Embedding(
            num_embeddings=num_lengths,
            embedding_dim=embedding_dim,
        )

    def forward(self, length_idx):
        embed = self.length_embedding(length_idx)
        return embed

class CategoryEmbeddings(nn.Module):
    def __init__(self, num_cats, embedding_dim):
        super(CategoryEmbeddings, self).__init__()

        self.cat_embedding = nn.Embedding(
            num_embeddings=num_cats,
            embedding_dim=embedding_dim,
        )

    def forward(self, cat_idx):
        embed = self.cat_embedding(cat_idx)
        return embed

# class DraughtEmbedding(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, embedding_dim),
#             nn.ReLU(),
#             nn.LayerNorm(embedding_dim)
#         )
#
#     def forward(self, draught):
#         if draught.dim() == 1:
#                 draught = draught.unsqueeze(-1)
#         return self.mlp(draught)
class DraughtEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, draught):
        # 支持标量、1D 或 2D 输入
        if draught.dim() == 0:
            draught = draught.unsqueeze(0).unsqueeze(-1)
        elif draught.dim() == 1:
            draught = draught.unsqueeze(-1)
        return self.mlp(draught)


# class CogEmbedding(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, embedding_dim),
#             nn.ReLU(),
#             nn.LayerNorm(embedding_dim)
#         )
#         self.mask_embedding = nn.Parameter(torch.randn(embedding_dim))
#
#     def forward(self, cog):
#         if cog.dim() == 1:
#                 cog = cog.unsqueeze(-1)
#         return self.mlp(cog)

# class SogEmbedding(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(1, embedding_dim),
#             nn.ReLU(),
#             nn.LayerNorm(embedding_dim)
#         )
#         self.mask_embedding = nn.Parameter(torch.randn(embedding_dim))
#
#     def forward(self, x):
#         """
#         x: [batch, 1] 或 [batch, seq_len, 1], 缺失值用 NaN 表示
#         """
#         mask = torch.isnan(x)  # [batch, 1] 或 [batch, seq_len, 1]
#         x = torch.where(mask, torch.tensor(0.0, device=x.device), x.float())
#         emb = self.mlp(x)
#         # 用 mask embedding 替换缺失值位置
#         emb[mask.expand_as(emb)] = self.mask_embedding
#         return emb
class SogEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
    def forward(self, x):
         emb = self.mlp(x)
         return emb

class CogEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # 输入维度是2（sin和cos）
        self.mlp = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )
    def forward(self, x):
        rad = x * math.pi / 180.0
        sin_x = torch.sin(rad)
        cos_x = torch.cos(rad)

        # 拼接 sin 和 cos，形成 2 维输入
        xy = torch.cat([sin_x, cos_x], dim=-1)  # [1, 2]

        # 送入 MLP 并返回 [embedding_dim]
        emb = self.mlp(xy)
        return emb


# class CoordEmbedding(nn.Module):
#     def __init__(self, embedding_dim):
#         super().__init__()
#         # 输入维度固定为2，因为是平面坐标 (x, y)
#         self.mlp = nn.Sequential(
#             nn.Linear(2, embedding_dim),
#             nn.ReLU(),
#             nn.LayerNorm(embedding_dim)
#         )
#         # mask embedding 用于替换缺失值
#         self.mask_embedding = nn.Parameter(torch.randn(embedding_dim))
#
#     def forward(self, xy_tensor):
#         """
#         xy_tensor: [batch, seq_len, 2] 或 [num_points, 2]
#         缺失值用 NaN 表示
#         """
#         # 判断缺失值，mask: True 表示缺失
#         mask = torch.isnan(xy_tensor).any(dim=-1)  # [batch, seq_len] 或 [num_points]
#
#         # 将 NaN 替换成0计算MLP
#         xy_tensor_clean = torch.where(mask.unsqueeze(-1),
#                                       torch.tensor(0.0, device=xy_tensor.device),
#                                       xy_tensor)
#
#         # 计算嵌入
#         if xy_tensor_clean.dim() == 2:
#             emb = self.mlp(xy_tensor_clean)
#             # 用 mask_embedding 替换缺失位置
#             emb[mask] = self.mask_embedding
#             return emb
#         elif xy_tensor_clean.dim() == 3:
#             batch, seq_len, _ = xy_tensor_clean.shape
#             xy_flat = xy_tensor_clean.reshape(-1, 2)
#             emb_flat = self.mlp(xy_flat)
#             emb = emb_flat.view(batch, seq_len, -1)
#             # 用 mask_embedding 替换缺失位置
#             emb[mask.unsqueeze(-1).expand(-1, -1, emb.shape[-1])] = self.mask_embedding
#             return emb
#         else:
#             raise ValueError("输入张量维度应为 2 或 3")
class CoordEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        # 输入维度固定为2，因为是平面坐标 (x, y)
        self.mlp = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, xy_tensor):

        if xy_tensor.dim() == 2:
            # 单条轨迹或点列表
            emb = self.mlp(xy_tensor)
            return emb
        elif xy_tensor.dim() == 3:
            # batch 处理
            batch, seq_len, _ = xy_tensor.shape
            xy_flat = xy_tensor.reshape(-1, 2)
            emb_flat = self.mlp(xy_flat)
            emb = emb_flat.view(batch, seq_len, -1)
            return emb
        else:
            raise ValueError("输入张量维度应为 2 或 3")

class FuseEmbeddingsMotion(nn.Module):
    def __init__(self, geometry_embed_dim, sog_embed_dim, cog_embed_dim, draught_embed_dim, time_embed_dim):
        super(FuseEmbeddingsMotion, self).__init__()
        embed_dim = geometry_embed_dim + sog_embed_dim + cog_embed_dim + draught_embed_dim + time_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim) #作用是对拼接后的嵌入进行特征变换，不改变维度（即保持 embed_dim 不变）
        self.leaky_relu = nn.LeakyReLU(0.2) #非线性激活，用于引入非线性特征，0.2 表示负半轴的斜率
        self.layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, geometry_embed, sog_embed, cog_embed, draught_embed, time_embed):
        x = self.fuse_embed(torch.cat((geometry_embed, sog_embed, cog_embed, draught_embed, time_embed), -1))
        x = self.leaky_relu(x)
        x = self.layer_norm(x)
        return x

class FuseEmbeddingsStatic(nn.Module):
    def __init__(self, mmsi_embed_dim, length_embed_dim, cat_embed_dim):
        super(FuseEmbeddingsStatic, self).__init__()
        embed_dim = mmsi_embed_dim + length_embed_dim + cat_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim) #作用是对拼接后的嵌入进行特征变换，不改变维度（即保持 embed_dim 不变）
        self.leaky_relu = nn.LeakyReLU(0.2) #非线性激活，用于引入非线性特征，0.2 表示负半轴的斜率
        self.layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, mmsi_embed, length_embed, cat_embed):
        x = self.fuse_embed(torch.cat(( mmsi_embed, length_embed, cat_embed), 0))
        x = self.leaky_relu(x)
        x = self.layer_norm(x)
        return x

class FuseEmbeddingsSpatialContext(nn.Module):
    def __init__(self, poi_embed_dim):
        super(FuseEmbeddingsSpatialContext, self).__init__()
        embed_dim = poi_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim) #作用是对拼接后的嵌入进行特征变换，不改变维度（即保持 embed_dim 不变）
        self.leaky_relu = nn.LeakyReLU(0.2) #非线性激活，用于引入非线性特征，0.2 表示负半轴的斜率
        self.layer_norm = nn.LayerNorm(embed_dim)
    def forward(self, start_ky_embed, end_ky_embed):
        x = self.fuse_embed(torch.cat(( start_ky_embed, end_ky_embed), 0))
        x = self.leaky_relu(x)
        x = self.layer_norm(x)
        return x

class FuseEmbeddings2(nn.Module):
    def __init__(self, user_embed_dim, poi_embed_dim):
        super(FuseEmbeddings2, self).__init__()
        embed_dim = user_embed_dim + poi_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, embed_dim) #作用是对拼接后的嵌入进行特征变换，不改变维度（即保持 embed_dim 不变）
        self.leaky_relu = nn.LeakyReLU(0.2) #非线性激活，用于引入非线性特征，0.2 表示负半轴的斜率

    def forward(self, user_embed, poi_embed):
        x = self.fuse_embed(torch.cat((user_embed, poi_embed), 0))
        x = self.leaky_relu(x)
        return x

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], 1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)


# class Time2Vec(nn.Module):
#     def __init__(self, activation, out_dim):
#         super(Time2Vec, self).__init__()
#         if activation == "sin":
#             self.l1 = SineActivation(1, out_dim)  #激活层，用于时间编码
#         elif activation == "cos":
#             self.l1 = CosineActivation(1, out_dim)
#
#     def forward(self, x):
#         x = self.l1(x)
#         return
# class Time2Vec(nn.Module):
#     def __init__(self, activation, out_dim):
#         super(Time2Vec, self).__init__()
#         if activation == "sin":
#             self.l1 = SineActivation(1, out_dim)
#         elif activation == "cos":
#             self.l1 = CosineActivation(1, out_dim)
#
#         self.out_dim = out_dim
#         # 定义一个可学习的 mask_embedding，用于缺失值位置
#         self.mask_embedding = nn.Parameter(torch.randn(out_dim))  # 形状为 [out_dim]
#
#     def forward(self, x):
#         # x: shape [batch_size, seq_len] 或 [batch_size, seq_len, 1]
#         mask = (x == -1)  # 假设 -1 是缺失值的占位符
#         x = torch.where(mask, torch.tensor(0.0, device=x.device), x)  # 替换缺失值为0计算嵌入
#
#         emb = self.l1(x)  # 正常的时间嵌入，形状为 [batch_size, seq_len, out_dim]
#
#         # 用 mask_embedding 替换缺失位置
#         if len(emb.shape) == 2:  # [batch, out_dim]
#             emb[mask] = self.mask_embedding
#         elif len(emb.shape) == 3:  # [batch, seq_len, out_dim]
#             emb[mask.unsqueeze(-1).expand(-1, -1, self.out_dim)] = self.mask_embedding
#
#         return emb

class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ReLU(),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, timestamps):
        # 支持单个标量、1D 或 2D 输入
        if timestamps.dim() == 0:
            timestamps = timestamps.view(1, 1)
        elif timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)
        return self.mlp(timestamps.float())




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# class TransformerModel(nn.Module):
#     def __init__(self, num_poi, num_cat, embed_size, nhead, nhid, nlayers, dropout=0.5):
#         super(TransformerModel, self).__init__()
#         from torch.nn import TransformerEncoder, TransformerEncoderLayer
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(embed_size, dropout) #Transformer 本身没有时间序列信息，因此使用 位置编码（Positional Encoding） 来补充位置信息，使模型能够理解序列顺序
#         encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout) #TransformerEncoderLayer（自带）
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers) #构造 Transformer 编码器（由多层 TransformerEncoderLayer 组成）
#         # self.encoder = nn.Embedding(num_poi, embed_size)
#         self.embed_size = embed_size
#         self.decoder_poi = nn.Linear(embed_size, num_poi)
#         self.decoder_time = nn.Linear(embed_size, 1)
#         self.decoder_cat = nn.Linear(embed_size, num_cat) # 解码层（预测层） 使用三种线性层进行预测，分别针对 POI、时间和类别。
#         self.init_weights()  #初始化部分参数权重，提升训练稳定性。
#
#     def generate_square_subsequent_mask(self, sz): #生成掩码，让当前时间步只能看到过去的时间步，防止信息泄露（Information Leak）
#         mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#         mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#         return mask
#
#     def init_weights(self): # 初始化解码层参数，保证网络收敛稳定
#         initrange = 0.1
#         self.decoder_poi.bias.data.zero_()
#         self.decoder_poi.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, src, src_key_padding_mask=None):
#         src = src * math.sqrt(self.embed_size)
#         src = self.pos_encoder(src)
#         x = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
#         out_poi = self.decoder_poi(x)
#         out_time = self.decoder_time(x)
#         out_cat = self.decoder_cat(x)
#         return out_poi, out_time, out_cat #刚删除的
class TransformerModel(nn.Module):
    def __init__(self, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'
        self.embed_size = embed_size

        # 位置编码
        self.pos_encoder = PositionalEncoding(embed_size, dropout)

        # Transformer 编码层
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # 三个解码头
        self.decoder_geometry = nn.Linear(embed_size, 2)  # 输出经纬度坐标 (x, y)
        self.decoder_sog = nn.Linear(embed_size, 1)       # 输出 SOG（速度）
        self.decoder_cog = nn.Linear(embed_size, 1)       # 输出 COG（航向）

        # 初始化权重
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        for decoder in [self.decoder_geometry, self.decoder_sog, self.decoder_cog]:
            if decoder.bias is not None:
                decoder.bias.data.zero_()
            decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_key_padding_mask=None):
        # 缩放 + 位置编码
        src = src * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)

        # Transformer 编码
        x = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)

        # 三路输出
        out_geo = self.decoder_geometry(x)
        out_sog = self.decoder_sog(x)
        out_cog = self.decoder_cog(x)

        return out_geo, out_sog, out_cog

#week
def weekday_t2v(weekday, f, out_features, w, b, w0, b0):
    if weekday.dim() == 1:
        weekday = weekday.unsqueeze(1)  # 保证输入为 [batch_size, 1]
    v1 = f(torch.matmul(weekday, w) + b)
    v2 = torch.matmul(weekday, w0) + b0
    return torch.cat([v1, v2], dim=1)

class SineWeekdayActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineWeekdayActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, weekday):
        return weekday_t2v(weekday, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineWeekdayActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineWeekdayActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, weekday):
        return weekday_t2v(weekday, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Weekday2Vec(nn.Module):
    def __init__(self, activation='sin', out_dim=8):
        super(Weekday2Vec, self).__init__()
        if activation == 'sin':
            self.embed = SineWeekdayActivation(1, out_dim)
        elif activation == 'cos':
            self.embed = CosineWeekdayActivation(1, out_dim)
        else:
            raise ValueError("activation must be 'sin' or 'cos'")

    def forward(self, weekday_tensor):
        # weekday_tensor shape: [batch_size] or [batch_size, 1]
        weekday_tensor = weekday_tensor.float()  # 转为float以进行正弦/余弦操作
        return self.embed(weekday_tensor)
