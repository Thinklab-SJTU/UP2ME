import torch
import torch.nn as nn
from ..pretrain_model.UP2ME_model import UP2ME_model
from .temporal_channel_layer import Temporal_Channel_Layer
from ..pretrain_model.embed import learnable_position_embedding
from .graph_structure import graph_construct
from einops import rearrange

class UP2ME_Detector(nn.Module):
    def __init__(self, pretrained_model_path, pretrain_args, finetune_flag=False, finetune_layers=3, dropout=0.0):
        super(UP2ME_Detector, self).__init__()
        self.pretrained_model_path = pretrained_model_path
        self.pretrain_args = pretrain_args
        self.data_dim = pretrain_args['data_dim']
        self.patch_size = pretrain_args['patch_size']

        self.finetune_flag = finetune_flag
        self.finetune_layers = finetune_layers
        self.dropout = dropout

        # load pre-trained model
        self.pretrained_model = UP2ME_model(
            data_dim=pretrain_args['data_dim'], patch_size=pretrain_args['patch_size'],\
            d_model=pretrain_args['d_model'], d_ff = pretrain_args['d_ff'], n_heads=pretrain_args['n_heads'], \
            e_layers=pretrain_args['e_layers'], d_layers = pretrain_args['d_layers'], dropout=pretrain_args['dropout'])
        self.load_pre_trained_model()
        
        # if fine-tune, add new layers
        if self.finetune_flag:
            self.enc_2_dec = nn.Linear(pretrain_args['d_model'], pretrain_args['d_model'])
            self.learnable_patch = nn.Parameter(torch.randn(pretrain_args['d_model']))
            self.position_embedding = learnable_position_embedding(pretrain_args['d_model'])
            self.channel_embedding = nn.Embedding(pretrain_args['data_dim'], pretrain_args['d_model'])
            self.init_enc2dec_param()
            
            self.temporal_channel_layers = nn.ModuleList()
            for _ in range(finetune_layers):
                self.temporal_channel_layers.append(Temporal_Channel_Layer(
                    d_model=pretrain_args['d_model'], n_heads=pretrain_args['n_heads'], d_ff=pretrain_args['d_ff'], dropout=dropout))
    
    def load_pre_trained_model(self):
        #load the pre-trained model
        pretrained_model = torch.load(self.pretrained_model_path, map_location='cpu')
        self.pretrained_model.load_state_dict(pretrained_model)
        
        #freeze the encoder and decoder of pre-trained model
        for param in self.pretrained_model.parameters():
            param.requires_grad = False     
    
    def init_enc2dec_param(self):
        self.position_embedding.load_state_dict(self.pretrained_model.position_embedding.state_dict())
        self.channel_embedding.load_state_dict(self.pretrained_model.channel_embedding.state_dict())
        self.learnable_patch.data.copy_(self.pretrained_model.learnable_patch.data)
        
        self.enc_2_dec.load_state_dict(self.pretrained_model.enc_2_dec.state_dict())   

    
    def train_mode(self):
        self.train()
        self.pretrained_model.eval()
        return
    
    def eval_mode(self):
        self.eval()
        return
    
    def immediate_detect(self, multi_ts):
        '''
        Immediate anomaly detection, the key idea is to iteratively mask one patch and use the remaining to reconstruct it
        It is more difficult to reconstruct abnormal patches than normal ones
        Args:
            multi_ts: [batch_size, ts_d, ts_len]
        '''

        batch_size, ts_d, ts_len = multi_ts.shape

        reconstruct_patches = []
        patch_num = ts_len // self.patch_size
        patch_idx = torch.arange(patch_num)[None, None, :].expand(batch_size, ts_d, -1).to(multi_ts.device)

        for i in range(patch_num):
            masked_patch_index = i * torch.ones([batch_size, ts_d, 1], dtype=torch.long).to(multi_ts.device)  # [batch_size, ts_d, 1]
            unmasked_patch_index = torch.cat([patch_idx[:, :, :i], patch_idx[:, :, i + 1:]], dim=-1)  # [batch_size, ts_d, patch_num-1]
            encoded_patch_unmasked = self.pretrained_model.encode_multi_to_patch(multi_ts, masked_patch_index, unmasked_patch_index)  # [batch_size, ts_d, patch_num-1, d_model]

            recons_full_ts = self.pretrained_model.decode_patch_to_multi(encoded_patch_unmasked, masked_patch_index, unmasked_patch_index)  # [batch_size, ts_d, ts_len]
            recons_patch = recons_full_ts[:, :, i * self.patch_size: (i + 1) * self.patch_size]  #pick out the  [batch_size, ts_d, patch_size]
            reconstruct_patches.append(recons_patch)

        reconstruct_ts = torch.cat(reconstruct_patches, dim=-1)  # [batch_size, ts_d, ts_len]

        return reconstruct_ts
    
    def forward(self, input_ts, neighbor_num = 10):        
        batch_size, ts_d, ts_len = input_ts.shape
        
        encoded_patch = self.pretrained_model.encode_multi_to_patch(input_ts)
        _, _, patch_num, d_model = encoded_patch.shape

        #compute the graph structure
        graph_adj = graph_construct(encoded_patch, k = neighbor_num)

        #<----------------------------------------------------prepare unmasked patches------------------------------------------------------->
        channel_idx =  torch.arange(ts_d).to(input_ts.device)
        channel_embed = self.channel_embedding(channel_idx) #[ts_d, d_model]

        masked_patch_embed = self.learnable_patch[None, None, None, :].expand(batch_size, ts_d, -1, -1)
        
        full_patch_idx = torch.arange(patch_num)[None, :].expand(batch_size, -1).to(input_ts.device)
        position_embed_masked = self.position_embedding([batch_size, patch_num, d_model], full_patch_idx) #[batch_size, patch_num, d_model]
        
        masked_patches_embed = channel_embed[None, :, None, :] + masked_patch_embed + position_embed_masked[:, None, :, :] #[batch_size, ts_d, patch_num, d_model]
        #<------------------------------------------------------------------------------------------------------------------------------------------>

        #iteratively mask each patch along time axes and use others to reconstruct it
        reconstruct_segments = []
        patch_idx = torch.arange(patch_num)[None, None, :].expand(batch_size, ts_d, -1).to(input_ts.device)

        for i in range(patch_num):
            masked_patch_idx = i * torch.ones([batch_size, ts_d, 1], dtype=torch.long).to(input_ts.device)
            unmasked_patch_idx = torch.cat([patch_idx[:, :, :i], patch_idx[:, :, i + 1:]], dim=-1)
            encoded_patch_unmasked = self.pretrained_model.encode_multi_to_patch(input_ts, masked_patch_idx, unmasked_patch_idx)
            encoded_patch_unmasked_transformed = self.enc_2_dec(encoded_patch_unmasked)
            patches_full = torch.cat([encoded_patch_unmasked_transformed[:, :, :i, :], masked_patches_embed[:, :, i:i+1, :], encoded_patch_unmasked_transformed[:, :, i:, :]], dim = -2) #[batch_size, ts_d, patch_num, d_model]

            #passing TC layers
            for layer in self.temporal_channel_layers:
                patches_full = layer(patches_full, graph_adj)
            
            #passing pretrained decoder
            flatten_patches_full = rearrange(patches_full, 'batch_size ts_d seq_len d_model -> (batch_size ts_d) seq_len d_model')
            flatten_channel_index = channel_idx[None, :].expand(batch_size, -1).reshape(-1)
            flatten_full_ts = self.pretrained_model.pretrain_decode(flatten_patches_full, flatten_channel_index) # (batch_size*ts_d, past_len+pred_len)
            full_ts = rearrange(flatten_full_ts, '(batch_size ts_d) ts_len -> batch_size ts_d ts_len', batch_size = batch_size)
            recons_segment = full_ts[:, :, i*self.patch_size: (i+1)*self.patch_size] #[batch_size, ts_d, patch_size]
            reconstruct_segments.append(recons_segment)
        
        reconstruct_ts = torch.cat(reconstruct_segments, dim=-1) #[batch_size, ts_d, ts_len]

        return reconstruct_ts