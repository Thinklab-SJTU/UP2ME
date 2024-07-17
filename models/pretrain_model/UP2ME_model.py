import torch
from torch import nn
from .embed import patch_embedding, learnable_position_embedding, RevIN
from .encoder_decoder import encoder, decoder
from einops import rearrange, repeat
from torch.nn.utils.rnn import pad_sequence
from loguru import logger

class UP2ME_model(nn.Module):
    def __init__(self, data_dim, patch_size,
                 d_model=256, d_ff=512, n_heads=4, e_layers=3, d_layers=1, dropout=0.0,
                 mask_ratio=0.75, device=torch.device('cuda:0')):
        super(UP2ME_model, self).__init__()

        self.data_dim = data_dim
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.dropout = dropout
        self.device = device

        self.RevIN = RevIN(data_dim, affine=True)
        self.patch_embedding = patch_embedding(patch_size, d_model)
        self.position_embedding = learnable_position_embedding(d_model)
        self.channel_embedding = nn.Embedding(data_dim, d_model)
        
        #encoder
        self.encoder = encoder(e_layers, d_model, n_heads, d_ff, dropout)
        
        #encoder-space to decoder-space
        self.enc_2_dec = nn.Linear(d_model, d_model)
        self.learnable_patch = nn.Parameter(torch.randn(d_model))
        
        #decoder
        self.decoder = decoder(patch_size, d_layers, d_model, n_heads, d_ff, dropout)
    
    def encode_uni_to_patch(self, ts, channel_idx, masked_patch_index=None, unmasked_patch_index=None, imputation_point_mask=None):
        '''
        Encode the unmaksed patches of unvariate time series to latent patches.

        Args:
            ts: time series with shape [batch_size, ts_length]
            channel_idx: channel index of time series with shape [batch_size]
            masked_patch_index: masked patch index with shape [batch_size, masked_patch_num]
            unmasked_patch_index: unmasked patch index with shape [batch_size, unmasked_patch_num]
            imputation_point_mask: point mask with shape [batch_size, ts_length], for imputation task.
        '''

        ts = self.RevIN(ts, channel_idx, mode='norm', mask=imputation_point_mask)

        patch_embed = self.patch_embedding.forward_uni(ts)  # [batch_size, patch_size, d_model]
        position_embed = self.position_embedding(patch_embed.shape)
        channel_embed = self.channel_embedding(channel_idx)  # [batch_size, d_model]
        patches = patch_embed + position_embed + channel_embed[:, None, :]  # [batch_size, patch_size, d_model]

        imputation_patch_mask = None
        if masked_patch_index is not None and unmasked_patch_index is not None:  # only encode the unmasked patches
            encoder_input = patches.gather(1, unmasked_patch_index[:, :, None].expand(-1, -1, self.d_model))
        elif imputation_point_mask is not None:
            imputation_patch_mask = rearrange(imputation_point_mask, 'batch_size (patch_num patch_size) -> batch_size patch_num patch_size', patch_size=self.patch_size)
            imputation_patch_mask = imputation_patch_mask.sum(dim=-1) > 0  # [batch_size, patch_num]
            encoder_input = patches
        else:
            encoder_input = patches
        encoded_patch_unmasked = self.encoder.forward_uni(encoder_input, imputation_patch_mask)  # [batch_size, unmasked_patch_num, d_model]

        return encoded_patch_unmasked
    
    def patch_concatenate(self, encoded_patch_unmasked, channel_idx, masked_patch_index, unmasked_patch_index):
        '''
        concatenate encoded unmasked patches and tokens indicating masked patches, i.e. First line in Equation (4) except enc-to-dec

        Args:
            encoded_patch_unmasked: encoded patches without masking with shape [batch_size, unmasked_patch_num, d_model]
            channel_idx: channel index of time series with shape [batch_size]
            masked_patch_index: masked patch index with shape [batch_size, masked_patch_num]
            unmasked_patch_index: unmasked patch index with shape [batch_size, unmasked_patch_num]  
        '''
        batch_size, unmasked_patch_num, _ = encoded_patch_unmasked.shape
        masked_patch_num = masked_patch_index.shape[1]

        patch_embed_masked = self.learnable_patch[None, None, :].expand(batch_size, masked_patch_num, -1)
        position_embed_masked = self.position_embedding(patch_embed_masked.shape, masked_patch_index)
        channel_embed_masked = self.channel_embedding(channel_idx)
        patches_masked = patch_embed_masked + position_embed_masked + channel_embed_masked[:, None, :]

        patches_full = torch.cat([patches_masked, encoded_patch_unmasked], dim=1)    #concate masked&unmasked patches
        patch_index_full = torch.cat([masked_patch_index, unmasked_patch_index], dim=1)
        origin_patch_index = torch.argsort(patch_index_full, dim=1)
        origin_patch_index = origin_patch_index.to(encoded_patch_unmasked.device)
        patches_full_sorted = patches_full.gather(1, origin_patch_index[:, :, None].expand(-1, -1, self.d_model)) #rearrange to original order

        return patches_full_sorted

    def pretrain_decode(self, full_patches, channel_idx):
        '''
        Decoding process, passing decoder and perform final projection

        Args:
            concated_patches: masked & unmasked patches [batch_size, total_patch_num, d_model]
            channel_idx: channel index of time series with shape [batch_size]
        '''

        reconstructed_ts = self.decoder.forward_uni(full_patches)
        reconstructed_ts = self.RevIN(reconstructed_ts, channel_idx, mode='denorm')

        return reconstructed_ts
    
    #some functions for downstream tasks
    def encode_multi_to_patch(self, multi_ts, masked_patch_index=None, unmasked_patch_index=None, imputation_point_mask=None):
        '''
        Encode the unmaksed patches of multivariate time series to latent patches.

        Args:
            multi_ts: time series with shape [batch_size, ts_d, ts_length]
            masked_patch_index: masked patch index with shape [batch_size, ts_d, masked_patch_num]
            unmasked_patch_index: unmasked patch index with shape [batch_size, ts_d, unmasked_patch_num]
            point_mask: point mask with shape [batch_size, ts_d, ts_length], for imputation task.
        '''

        batch_size, ts_d, ts_length = multi_ts.shape

        ts_flatten = rearrange(multi_ts, 'batch_size ts_d ts_length -> (batch_size ts_d) ts_length')
        channel_idx = torch.arange(self.data_dim)[None, :].expand(batch_size, -1).to(multi_ts.device)
        channel_flatten = rearrange(channel_idx, 'batch_size ts_d -> (batch_size ts_d)')

        if masked_patch_index is not None and unmasked_patch_index is not None:  # only encode the unmasked patches
            masked_patch_flatten = rearrange(masked_patch_index, 'batch_size ts_d masked_patch_num -> (batch_size ts_d) masked_patch_num')
            unmasked_patch_flatten = rearrange(unmasked_patch_index, 'batch_size ts_d unmasked_patch_num -> (batch_size ts_d) unmasked_patch_num')
        else:
            masked_patch_flatten, unmasked_patch_flatten = None, None

        if imputation_point_mask is not None:
            imputation_point_mask_flatten = rearrange(imputation_point_mask, 'batch_size ts_d ts_length -> (batch_size ts_d) ts_length')
        else:
            imputation_point_mask_flatten = None
        
        encoded_patch_flatten = self.encode_uni_to_patch(ts_flatten, channel_flatten, masked_patch_flatten, unmasked_patch_flatten, imputation_point_mask_flatten)
        encoded_patch = rearrange(encoded_patch_flatten, '(batch_size ts_d) patch_num d_model -> batch_size ts_d patch_num d_model', batch_size=batch_size)

        return encoded_patch
    
    def decode_patch_to_multi(self, encoded_patch_unmasked, masked_patch_index, unmasked_patch_index):
        batch_size, ts_d, unmasked_patch_num, _ = encoded_patch_unmasked.shape

        flatten_encoded_patch_unmasked = rearrange(encoded_patch_unmasked, 'batch_size ts_d unmasked_patch_num d_model -> (batch_size ts_d) unmasked_patch_num d_model')
        flatten_masked_patch_index = rearrange(masked_patch_index, 'batch_size ts_d masked_patch_num -> (batch_size ts_d) masked_patch_num')
        flatten_unmasked_patch_index = rearrange(unmasked_patch_index, 'batch_size ts_d unmasked_patch_num -> (batch_size ts_d) unmasked_patch_num')
        flatten_channel_idx = torch.arange(self.data_dim)[None, :].expand(batch_size, -1).reshape(batch_size * ts_d).to(encoded_patch_unmasked.device)
        
        flatten_encoded_patch_unmasked = self.enc_2_dec(flatten_encoded_patch_unmasked)
        flatten_full_patch = self.patch_concatenate(flatten_encoded_patch_unmasked, flatten_channel_idx, flatten_masked_patch_index, flatten_unmasked_patch_index)
        flatten_reconstructed_ts = self.pretrain_decode(flatten_full_patch, flatten_channel_idx)

        reconstructed_ts = rearrange(flatten_reconstructed_ts, '(batch_size ts_d) ts_len -> batch_size ts_d ts_len', batch_size=batch_size)

        return reconstructed_ts
    
    
    # imputation-related functions
    def decode_CDpatch_to_multi(self, pacthes_with_CD):
        '''
        input the patches with cross-channel dependency (in finetune), and output the reconstructed full time series
        patches_with_CD: [batch_size, ts_d, patch_num, d_model]
        '''
        batch_size, ts_d, patch_num, _ = pacthes_with_CD.shape
        decode_ts = self.decoder.forward_multi(pacthes_with_CD)  # [batch_size, ts_d, ts_len]
        channel_idx_flatten = torch.arange(self.data_dim)[None, :].expand(batch_size, -1).reshape(batch_size * ts_d).to(decode_ts.device)
        decode_ts_flatten = rearrange(decode_ts, 'batch_size ts_d ts_len -> (batch_size ts_d) ts_len')
        decode_ts_flatten = self.RevIN(decode_ts_flatten, channel_idx_flatten, mode='denorm')
        decode_ts = rearrange(decode_ts_flatten, '(batch_size ts_d) ts_len -> batch_size ts_d ts_len', batch_size=batch_size)

        return decode_ts

    def encode_masked_ts(self, ts, mask, channel_idx):
        '''
        Encode the unmasked patches of unvariate varational time series to latent patches.

        Args:
            ts: flattened time series with shape [batch_size, ts_length]
            mask: times series with missing data's mask [batch_size, ts_length]
            channel_idx: channel index of time series with shape [batch_size]
        '''
        batch_size, ts_d, ts_len = ts.shape
        ts = ts.reshape(-1, ts_len)
        mask = mask.reshape(-1, ts_len)

        channel_idx = channel_idx.reshape(-1)
        ts = self.RevIN(ts, channel_idx, 'norm', mask)

        patch_embed = self.patch_embedding.forward_uni(ts)  # [batch_size, patch_num, d_model]
        position_embed = self.position_embedding(patch_embed.shape)

        # channel_idx = channel_idx.reshape(batch_size, ts_d)
        channel_embed = self.channel_embedding(channel_idx)  # [batch_size, d_model]

        patches = patch_embed + position_embed + channel_embed[:, None, :]  # [batch_size, patch_num, d_model]

        patch_mask = rearrange(mask, 'batch_size (patch_num patch_size) -> batch_size patch_num patch_size', patch_size=self.patch_size)
        patch_mask = patch_mask.sum(dim=-1) > 0  # [batch_size, patch_num]

        unmasked_pacthes_encoded = self.encoder.forward_uni(patches, patch_mask)  # [batch_size, padded_unmasked_patch_num, d_model]
        return unmasked_pacthes_encoded, patch_mask

    def decode_masked_ts(self, unmasked_patches_encoded, channel_idx, patch_mask, temporal_geometric_layer=None):
        '''
        Decoder for variational multivariate series: [batch_size, padded_unmasked_patch_num, d_model] 
        '''
        batch_size, patch_num, _ = unmasked_patches_encoded.shape
        unmasked_patches_encoded = self.enc_2_dec(unmasked_patches_encoded)

        patch_embed_masked = self.learnable_patch[None, None, :].expand(batch_size, patch_num, -1)  # [batch_size, patch_num, d_model]
        position_embed_masked = self.position_embedding(patch_embed_masked.shape)  # [batch_size, patch_num, d_model]
        channel_embed_masked = self.channel_embedding(channel_idx)
        patches_masked = patch_embed_masked + position_embed_masked + channel_embed_masked[:, None, :]

        full_patches = patches_masked * patch_mask[:, :, None] + unmasked_patches_encoded * (~patch_mask[:, :, None])  # [batch_size, patch_num, d_model]

        if temporal_geometric_layer is not None:
            full_patches = temporal_geometric_layer(full_patches)

        reconstructed_ts = self.decoder.forward_uni(full_patches)

        reconstructed_ts = self.RevIN(reconstructed_ts, channel_idx, mode='denorm')
        reconstructed_ts = reconstructed_ts.reshape(batch_size // self.data_dim, self.data_dim, -1)
        return reconstructed_ts


