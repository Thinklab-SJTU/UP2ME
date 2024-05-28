import torch
import math
import numpy as np


def batch_cosine_similarity(x, y, eps=1e-8):
    '''
    compute the cosine similarity matrix among variables and get D * D matrix
    x, y: [batch_size, ts_d, d_model]
    '''

    inner_dot = torch.einsum('bqd,bkd->bqk', x, y)
    x_norm = torch.norm(x, dim=-1, keepdim=True)  # [batch_size, ts_d, 1]
    y_norm = torch.norm(y, dim=-1, keepdim=True)  # [batch_size, ts_d, 1]
    norm_dot = torch.einsum('bqd,bkd->bqk', x_norm, y_norm)  # [batch_size, ts_d, ts_d]
    cos_corr = inner_dot / (norm_dot + eps)

    return cos_corr

def k_nearest_neighbor(corr_matrix, k=10):
    '''
    return an adjacency matrix of the k nearest neighbors
    corr_matrix: [batch_size, ts_d, ts_d]
    '''
    batch_size, ts_d, _ = corr_matrix.shape
    edges_knn = torch.topk(corr_matrix, k, dim=-1)[1]  # [batch_size * ts_d, k]

    knn_adj = torch.zeros(batch_size, ts_d, ts_d).to(corr_matrix.device)
    knn_adj.scatter_(-1, edges_knn, 1)
    knn_adj = knn_adj.permute(0, 2, 1)  # source to target

    return knn_adj


def graph_construct(encoded_patch, patch_mask=None, k=10):
    # encoded_patch: [batch_size, ts_d, patch_num, d_model]
    # patch_mask: [batch_size, ts_d, patch_num], 1 for masked, 0 for unmasked
    batch_size, ts_d, patch_num, d_model = encoded_patch.shape

    if patch_mask is not None:
        encoded_patch = encoded_patch.masked_fill(patch_mask[:, :, :, None] == 1, -np.inf)

    channel_encode = encoded_patch.max(dim=-2)[0]  # max pooling over the patch dimension, [batch_size, ts_d, d_model]
    corr_matrix = batch_cosine_similarity(channel_encode, channel_encode)  # [batch_size, ts_d, ts_d]

    # KNN graph
    knn_adj = k_nearest_neighbor(corr_matrix, k)  # [batch_size, ts_d, ts_d] i-->j

    # top k*N graph
    top_k_threshold = torch.topk(corr_matrix.reshape(batch_size, -1), k * ts_d, dim=-1)[0][:, -1]  # [batch_size]
    top_k_adj = (corr_matrix >= top_k_threshold[:, None, None]).long()  # [batch_size, ts_d, ts_d] i-->j

    graph_adj = knn_adj * top_k_adj  # consider both knn and top-k threshold, [batch_size, ts_d, ts_d] i-->j

    return graph_adj
