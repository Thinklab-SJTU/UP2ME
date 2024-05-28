import torch
import numpy as np
import math
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.utils import to_dense_batch
from einops import rearrange

def batch_to_sparse(batch_node, adj_matrix):
    '''
    convert the batched node features and adjacency matrix to sparse tensor for pytorch geometric
    batch_node: [batch_size, node_num, d_model]
    adj_matrix: [batch_size, node_num, node_num], binary adjacency matrix
    '''
    batch_size, node_num, _ = batch_node.shape
    offset, row, col = torch.nonzero(adj_matrix > 0).t() # [edge_num]

    row = row + offset * node_num
    col = col + offset * node_num
    edge_index = torch.stack([row, col], dim=0).long()
    list_node = batch_node.reshape(batch_size * node_num, -1)
    batch_idx= torch.arange(0, batch_size, device=batch_node.device)[:, None].expand(-1, node_num).reshape(-1)

    return list_node, edge_index, batch_idx

class Graph_MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super(Graph_MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout

        self.MHA_layer = TransformerConv(in_channels=d_model, out_channels=self.head_dim, heads=n_heads, concat=True, dropout=dropout)
        self.out_projection = nn.Linear(self.n_heads * self.head_dim, d_model)
    
    def forward(self, node, edge_index):
        # node: [|V|, d_model]
        # edge_index: [2, |E|]
        # batch_idx: [|V|]

        node = self.MHA_layer(node, edge_index) # [|V|, n_heads * head_dim]

        output = self.out_projection(node) # [|V|, d_model]

        return output

class Graph_TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(Graph_TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.attention = Graph_MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, node, edge_index, norm_first=False):
        
        if norm_first:
            node = node + self._sa_block(self.norm1(node), edge_index)
            node = node + self._ff_block(self.norm2(node))
        else:
            node = self.norm1(node + self._sa_block(node, edge_index))
            node = self.norm2(node + self._ff_block(node))

        return node
    
    def _sa_block(self, node, edge_index):
        node = self.attention(node, edge_index)
        return self.dropout1(node)

    # feed forward block
    def _ff_block(self, node):
        node = self.feed_forward(node)
        return self.dropout2(node)

class Temporal_Channel_Layer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(Temporal_Channel_Layer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        self.temporal_layer_blocks = 1
        self.temporal_layers = nn.ModuleList()
        for i in range(self.temporal_layer_blocks):
            self.temporal_layers.append(TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first = True))
        self.channel_layer_blocks = 1
        self.channel_layers = nn.ModuleList()
        for i in range(self.channel_layer_blocks):
            self.channel_layers.append(Graph_TransformerEncoderLayer(d_model, n_heads, d_ff, dropout))

    def forward(self, x, graph_adj):
        # x: [batch_size, ts_d, seq_len, d_model]
        # graph_adj: [batch_size, ts_d, ts_d]
        
        batch_size, ts_d, seq_len, d_model = x.shape
        temporal_input = x.reshape(batch_size * ts_d, seq_len, d_model)
        temporal_output = temporal_input
        for temporal_layer in self.temporal_layers:
            temporal_output = temporal_layer(temporal_output) # [batch_size * ts_d, seq_len, d_model]

        channel_input = rearrange(temporal_output, '(batch_size ts_d) seq_len d_model -> (batch_size seq_len) ts_d d_model', batch_size = batch_size)
        graph_adj_expand = graph_adj[:, None, :, :].expand(-1, seq_len, -1, -1)
        graph_adj_expand = rearrange(graph_adj_expand, 'batch_size seq_len ts_d1 ts_d2 -> (batch_size seq_len) ts_d1 ts_d2')

        channel_input_sparse, edge_index, batch_idx = batch_to_sparse(channel_input, graph_adj_expand) #for torch geometric, [batch_size * seq_len * ts_d, d_model], [2, edge_num], [batch_size * seq_len * ts_d]
        channel_output_sparse = channel_input_sparse

        for channel_layer in self.channel_layers:
            channel_output_sparse = channel_layer(channel_output_sparse, edge_index)
        channel_output_batch = to_dense_batch(channel_output_sparse, batch=batch_idx)[0] # [batch_size * seq_len, ts_d, d_model]

        output = rearrange(channel_output_batch, '(batch_size seq_len) ts_d d_model -> batch_size ts_d seq_len d_model', batch_size = batch_size)

        return output
