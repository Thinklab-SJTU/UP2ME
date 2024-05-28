import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class patch_embedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super(patch_embedding, self).__init__()
        self.patch_size = patch_size
        self.linear = nn.Linear(patch_size, d_model, bias = False)
    
    def forward_uni(self, x):
        #the univariate forward mode

        batch_size, ts_len = x.shape

        x_patch = rearrange(x, 'b (patch_num patch_size) -> b patch_num patch_size', patch_size = self.patch_size)
        x_embed = self.linear(x_patch) #[batch_size, patch_num, d_model]
        
        return x_embed
    
    def forward_multi(self, x):
        #the multivariate forward mode

        batch_size, ts_len, ts_dim = x.shape

        x_patch = rearrange(x, 'b (patch_num patch_size) d -> b d patch_num patch_size', patch_size = self.patch_size)
        x_embed = self.linear(x_patch) #[batch_size, ts_dim, patch_num, d_model]

        return x_embed

class learnable_position_embedding(nn.Module):
    def __init__(self, d_model, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model), requires_grad=True)

    def forward(self, patch_shape, index=None):
        """Positional encoding

        Args:
            patch_shape: shape of time series, should be [batch_size, patch_num, d_model] or [batch_size, ts_d, patch_num, d_model]
            index (list or None): add positional embedding by index, [batch_size, patch_num] or [batch_size, ts_d, patch_num]

        Returns:
            torch.tensor: output sequence
        """
        if (len(patch_shape) == 3):
            #for univariate time series
            batch_size, patch_num, _ = patch_shape
            position_embedding_expand = self.position_embedding[None, :, :].expand(batch_size, -1, -1)
            if index is None:
                pe = position_embedding_expand[:, :patch_num, :]  #not assigned, 0 ~ patch_num - 1
            else:
                index_expand = index[:, :, None].expand(-1, -1, self.d_model)
                pe = position_embedding_expand.gather(1, index_expand)
            return pe #[batch_size, patch_num, d_model]
        
        elif (len(patch_shape) == 4):
            #for multivariate time series
            batch_size, ts_d, patch_num, _ = patch_shape
            position_embedding_expand = self.position_embedding[None, None, :, :].expand(batch_size, ts_d, -1, -1)
            if index is None:
                pe = position_embedding_expand[:, :, :patch_num, :]
            else:
                index_expand = index[:, :, :, None].expand(-1, -1, -1, self.d_model)
                pe = position_embedding_expand.gather(2, index_expand)
            return pe #[batch_size, ts_d, patch_num, d_model]


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-7, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, channel_idx, mode: str, mask=None):
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x, channel_idx)
        elif mode == 'denorm':
            x = self._denormalize(x, channel_idx)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))

    def _get_statistics(self, x, mask):
        '''
            x: time series with shape [batch_size, ts_length]
            channel_idx: channel index of time series with shape [batch_size]
            mask: binary mask with the same shape as x, 1 for non-sense values to be masked
        '''
        if mask is not None:
            masked_x = x.masked_fill(mask == 1, 0)
            self.mean = (torch.sum(masked_x, dim=-1, keepdim=True) / torch.sum(mask == 0, dim=-1, keepdim=True)).detach()
            diff = masked_x - self.mean
            masked_diff = diff.masked_fill(mask == 1, 0)
            self.stdev = (torch.sqrt(torch.sum(masked_diff * masked_diff, dim=-1, keepdim=True) / torch.sum(mask == 0, dim=-1, keepdim=True) + self.eps)).detach()
        else:
            self.mean = torch.mean(x, dim=-1, keepdim=True).detach()  # [batch_size, 1]
            self.stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + self.eps).detach()  # [batch_size, 1]

    def _normalize(self, x, channel_idx):
        '''
        channel_idx: [batch_size]
        x: [batch_size, ts_length]
        '''        
        x = x - self.mean
        x = x / self.stdev

        if self.affine:
            affine_weight = self.affine_weight.gather(0, channel_idx)  # [batch_size]
            affine_bias = self.affine_bias.gather(0, channel_idx)  # [batch_size]

            x = x * affine_weight[:, None]
            x = x + affine_bias[:, None]

        return x

    def _denormalize(self, x, channel_idx):
        if self.affine:
            affine_weight = self.affine_weight.gather(0, channel_idx)  # [batch_size]
            affine_bias = self.affine_bias.gather(0, channel_idx)  # [batch_size]

            x = x - affine_bias[:, None]
            x = x / (affine_weight[:, None] + self.eps * self.eps)

        x = x * self.stdev
        x = x + self.mean

        return x