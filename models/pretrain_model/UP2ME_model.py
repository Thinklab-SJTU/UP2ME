import torch
from torch import nn
from .embed import patch_embedding, learnable_position_embedding, RevIN
from .mask_gen import MaskGenerator
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

        self.mask_generator = MaskGenerator(mask_ratio)

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
    
    def pretrain_encode(self, ts, channel_idx):
        '''
        Encoding process for pretraining, input a batch of univariate time series, generate mask, and output the encoded unmasked patches.

        Args:
            ts: time series with shape [batch_size, ts_length]
            channel_idx: channel index of time series with shape [batch_size]
        '''

        batch_size, ts_length = ts.shape
        patch_num = ts_length // self.patch_size

        masked_patch_index, unmasked_patch_index = self.mask_generator.forward_uni([batch_size, patch_num, self.d_model])
        masked_patch_index = masked_patch_index.to(ts.device)
        unmasked_patch_index = unmasked_patch_index.to(ts.device)

        encoded_patch_unmasked = self.encode_uni_to_patch(ts, channel_idx, masked_patch_index, unmasked_patch_index)

        return encoded_patch_unmasked, masked_patch_index, unmasked_patch_index
    
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
    
    def get_reconstructed_masked_ts_uni(self, reconstruction_full, target_full, masked_patch_index):
        # pick out the masked patch to compute the loss, i.e. only compute loss on masked patches

        batch_size, ts_length = target_full.shape
        target_patches = rearrange(target_full, 'batch_size (patch_num patch_size) -> batch_size patch_num patch_size', patch_size=self.patch_size)
        masked_patch_index = masked_patch_index.to(target_full.device)
        masked_target_patches = target_patches.gather(1, masked_patch_index[:, :, None].expand(-1, -1, self.patch_size))  # [batch_size, masked_patch_num, patch_size]

        reconstruction_patches = rearrange(reconstruction_full, 'batch_size (patch_num patch_size) -> batch_size patch_num patch_size', patch_size=self.patch_size)
        masked_reconstruction_patches = reconstruction_patches.gather(1, masked_patch_index[:, :, None].expand(-1, -1, self.patch_size))  # [batch_size, masked_patch_num, patch_size]

        return masked_reconstruction_patches, masked_target_patches # [batch_size, masked_patch_num, patch_size]
    
    def pretrain_forward(self, ts, channel_idx):
        """Forward process for pretraining, input a batch of univariate time series, perform masking and output the reconstructed masked patches. 

        Args:
            ts: time series with shape [batch_size, ts_length]
            channel_idx: channel index of time series with shape [batch_size]
        """
        
        #encode (Equation 2 and 3)
        encoded_patch_unmasked, masked_patch_index, unmasked_patch_index = self.pretrain_encode(ts, channel_idx)
        
        #decode (Equation 4)
        encoded_patch_unmasked = self.enc_2_dec(encoded_patch_unmasked) #Equation 4, line 1.1
        full_patch = self.patch_concatenate(encoded_patch_unmasked, channel_idx, masked_patch_index, unmasked_patch_index) #Equation 4, line 1.2
        reconstructed_ts = self.pretrain_decode(full_patch, channel_idx) #Equation 4, line 2&3
        
        #get pairs (Equation 5)
        masked_reconstruction_patches, masked_target_patches = self.get_reconstructed_masked_ts_uni(reconstructed_ts, ts, masked_patch_index) 

        return masked_reconstruction_patches, masked_target_patches
    

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
    
