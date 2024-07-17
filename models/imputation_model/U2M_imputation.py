import torch
import torch.nn as nn


from models.pretrain_model.UP2ME_model import UP2ME_model
from models.pretrain_model.embed import learnable_position_embedding
from models.finetune_model.temporal_channel_layer import Temporal_Channel_Layer
from models.finetune_model.graph_structure import graph_construct
from einops import rearrange
from loguru import logger


class U2M_imputation(nn.Module):
    def __init__(self, pretrained_model_path, args, pretrain_args, gnn_type='GraphTransformer', finetune_layers=3, dropout=0.0):
        super(U2M_imputation, self).__init__()
        self.pretrained_model_path = pretrained_model_path
        self.pretrain_args = pretrain_args
        self.imputation_layers = finetune_layers
        self.args = args

        # iniitalize the pretrain and forecast model
        self.pretrain_model = UP2ME_model(
            data_dim=pretrain_args['data_dim'], patch_size=pretrain_args['patch_size'],
            d_model=pretrain_args['d_model'], d_ff=pretrain_args['d_ff'], n_heads=pretrain_args['n_heads'],
            e_layers=pretrain_args['e_layers'], d_layers=pretrain_args['d_layers'], dropout=pretrain_args['dropout'])

        self.d_model = pretrain_args['d_model']
        self.patch_size = pretrain_args['patch_size']
        self.in_len = self.args.in_len
        self.data_dim = pretrain_args['data_dim']

        self.enc_2_dec = nn.Linear(pretrain_args['d_model'], pretrain_args['d_model'])
        self.learnable_patch = nn.Parameter(torch.randn(pretrain_args['d_model']))
        self.position_embedding = learnable_position_embedding(pretrain_args['d_model'])
        self.channel_embedding = nn.Embedding(pretrain_args['data_dim'], pretrain_args['d_model'])

        self.temporal_variable_layers = nn.ModuleList()
        for i in range(self.imputation_layers):
            self.temporal_variable_layers.append(Temporal_Channel_Layer(
                d_model=pretrain_args['d_model'], n_heads=pretrain_args['n_heads'], d_ff=pretrain_args['d_ff'], dropout=dropout))

        # load pre-trained model
        self.load_pre_trained_model()

    def load_pre_trained_model(self):
        # load the pre-trained model
        pretrain_model = torch.load(
            self.pretrained_model_path, map_location='cpu')

        # initialize the embedding module of forecast model with that of pre-trained model
        self.position_embedding.load_state_dict(self.pretrain_model.position_embedding.state_dict())
        self.channel_embedding.load_state_dict(self.pretrain_model.channel_embedding.state_dict())
        self.learnable_patch.data.copy_(self.pretrain_model.learnable_patch.data)
        self.enc_2_dec.load_state_dict(self.pretrain_model.enc_2_dec.state_dict())

        self.pretrain_model.load_state_dict(pretrain_model)
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

    def train_mode(self):
        self.train()
        self.pretrain_model.eval()
        return

    def eval_mode(self):
        self.eval()
        return

    def forward_pretrain(self, ts, mask):

        batch_size, ts_d, ts_len = ts.shape

        channel_idx = torch.arange(self.data_dim)[None, :].expand(batch_size, -1).reshape(batch_size * ts_d).to(ts.device)

        unmasked_patches_encoded, patch_mask = self.pretrain_model.encode_masked_ts(ts, mask, channel_idx)

        reconstructed_ts = self.pretrain_model.decode_masked_ts(unmasked_patches_encoded, channel_idx, patch_mask)
        return reconstructed_ts

    def forward_finetune(self, ts, mask, neighbor_num):
        batch_size, ts_d, ts_len = ts.shape
        assert self.data_dim == ts_d, f"ts_d:{ts_d}, data_dim:{self.data_dim}"
        channel_idx = torch.arange(self.data_dim)[None, :].expand(batch_size, -1).to(ts.device)

        unmasked_patches_encoded, patch_mask = self.pretrain_model.encode_masked_ts(ts, mask, channel_idx)  # [batch_size, patch_num, d_model]
        assert self.args.is_training

        batch_size, patch_num, _ = unmasked_patches_encoded.shape

        # compute the graph structure
        unmasked_patches_encoded = rearrange(unmasked_patches_encoded, '(batch_size ts_d) patch_num patch_size -> batch_size ts_d patch_num patch_size', ts_d=ts_d)  # [batch_size, ts_d, patch_num, d_model]
        patch_mask = rearrange(patch_mask, '(batch_size ts_d) patch_num -> batch_size ts_d patch_num', ts_d=ts_d)  # [batch_size, ts_d, patch_num, d_model]
        graph_adj = graph_construct(unmasked_patches_encoded, patch_mask=patch_mask, k=neighbor_num)  # [batch_size, ts_d, patch_num, d_model]

        unmasked_patches_encoded = self.enc_2_dec(unmasked_patches_encoded)

        patch_embed_masked = self.learnable_patch[None, None, None, :].expand(batch_size // ts_d, ts_d, patch_num, -1)  # [batch_size, ts_d, patch_num, d_model]
        position_embed_masked = self.position_embedding(patch_embed_masked.shape)  # [batch_size, ts_d, patch_num, d_model]

        channel_idx = channel_idx.view(-1, ts_d)  # [batch_size, ts_d]
        channel_embed_masked = self.channel_embedding(channel_idx)  # [batch_size, ts_d, d_model]
        patches_masked = patch_embed_masked + position_embed_masked + channel_embed_masked[:, :, None, :]

        patches_with_CD = torch.where(patch_mask[:, :, :, None], patches_masked, unmasked_patches_encoded)  # [batch_size, ts_d, patch_num, d_model]

        # capture temporal and variable dependency
        if self.args.gnn_type == "pretrain_model":
            pass
        else:
            for layer in self.temporal_variable_layers:
                patches_with_CD = layer(patches_with_CD, graph_adj)

        patches_with_CD = self.pretrain_model.decode_CDpatch_to_multi(patches_with_CD)  # [batch_size, ts_d, ts_len]

        return patches_with_CD
