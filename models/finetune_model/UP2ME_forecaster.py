import torch
import torch.nn as nn
from ..pretrain_model.UP2ME_model import UP2ME_model
from .temporal_channel_layer import Temporal_Channel_Layer
from ..pretrain_model.embed import learnable_position_embedding
from .graph_structure import graph_construct
from einops import rearrange

class UP2ME_forecaster(nn.Module):
    def __init__(self, pretrained_model_path, pretrain_args, finetune_flag=False, finetune_layers=3, dropout=0.0):
        super(UP2ME_forecaster, self).__init__()
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
    
    def immediate_forecast(self, multi_ts, pred_len):
        '''
        Immediate reaction mode, directly use the pretrained model to perform multi-variate forecasting without any parameter modification
        Args:
            multi_ts: [batch_size, ts_d, past_len]
            pred_len: [batch_size]
        '''

        batch_size, ts_d, past_len = multi_ts.shape
        
        # encode past patches
        encoded_past_patch = self.pretrained_model.encode_multi_to_patch(multi_ts)
        
        #prepare masked and unmasked indices
        full_len = past_len + pred_len
        full_patch_num = full_len // self.patch_size
        past_patch_num = past_len // self.patch_size
        full_patch_idx = torch.arange(full_patch_num)[None, None, :].expand(batch_size, ts_d, -1).to(multi_ts.device)
        past_patch_idx = full_patch_idx[:, :, :past_patch_num]
        pred_patch_idx = full_patch_idx[:, :, past_patch_num:]

        reconstructed_full_ts = self.pretrained_model.decode_patch_to_multi(encoded_past_patch, pred_patch_idx, past_patch_idx)

        pred_ts = reconstructed_full_ts[:, :, past_len:]

        return pred_ts

    def forward(self, past_ts, pred_patch_num, neighbor_num = 10):
        batch_size, ts_d, _ = past_ts.shape

        encoded_patch_past = self.pretrained_model.encode_multi_to_patch(past_ts)
        
        #compute the graph structure
        graph_adj = graph_construct(encoded_patch_past, k = neighbor_num)
        
        encoded_patch_past_transformed = self.enc_2_dec(encoded_patch_past) #[batch_size, ts_d, patch_num, d_model]

        #<----------------------------------concatenate past and future patches-------------------------------------------->
        _, _, past_patch_num, d_model = encoded_patch_past_transformed.shape
        channel_idx =  torch.arange(ts_d).to(past_ts.device)
        channel_embed = self.channel_embedding(channel_idx) #[ts_d, d_model]
        channel_embed_future = channel_embed[None, :, None, :].expand(batch_size, -1, pred_patch_num, -1) #[batch_size, ts_d, pred_patch_num, d_model]

        patch_embed_future = self.learnable_patch[None, None, None, :].expand(batch_size, ts_d, pred_patch_num, -1)
        
        future_patch_idx = torch.arange(past_patch_num, past_patch_num + pred_patch_num)[None, :].expand(batch_size, -1).to(past_ts.device)
        position_embed_future = self.position_embedding([batch_size, pred_patch_num, d_model], future_patch_idx) #[batch_size, pred_patch_num, d_model]
        position_embed_future = position_embed_future[:, None, :, :].expand(-1, ts_d, -1, -1) #[batch_size, ts_d, pred_patch_num, d_model]
        
        patches_future = patch_embed_future + position_embed_future + channel_embed_future #[batch_size, ts_d, pred_patch_num, d_model]
        patches_past = encoded_patch_past_transformed
        patches_full = torch.cat((patches_past, patches_future), dim = -2) #[batch_size, ts_d, past_patch_num+pred_patch_num, d_model]
        #<----------------------------------------------------------------------------------------------------------------->

        #passing TC layers
        for layer in self.temporal_channel_layers:
            patches_full = layer(patches_full, graph_adj)
        
        #passing pretrained decoder
        flatten_patches_full = rearrange(patches_full, 'batch_size ts_d seq_len d_model -> (batch_size ts_d) seq_len d_model')
        flatten_channel_index = channel_idx[None, :].expand(batch_size, -1).reshape(-1)
        flatten_full_ts = self.pretrained_model.pretrain_decode(flatten_patches_full, flatten_channel_index) # (batch_size*ts_d, past_len+pred_len)
        full_ts = rearrange(flatten_full_ts, '(batch_size ts_d) ts_len -> batch_size ts_d ts_len', batch_size = batch_size)
        pred_ts = full_ts[:, :, -pred_patch_num*self.pretrain_args['patch_size']:]

        return pred_ts