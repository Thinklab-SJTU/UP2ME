import torch
from torch import nn

class MaskGenerator(nn.Module):
    """Mask generator."""

    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward_uni(self, patch_shape):
        batch_size, patch_num, _ = patch_shape
        mask_num = int(patch_num * self.mask_ratio)

        rand_tmp = torch.rand(batch_size, patch_num)
        random_idx = torch.argsort(rand_tmp, dim = -1)
        masked_idxs = random_idx[:, :mask_num].sort(dim = -1)[0]
        unmasked_idxs = random_idx[:, mask_num:].sort(dim = -1)[0]

        return masked_idxs, unmasked_idxs
    
    def forward_multi(self, patch_shape):
        batch_size, ts_d, patch_num, _ = patch_shape
        mask_num = int(patch_num * self.mask_ratio)

        rand_tmp = torch.rand(batch_size, ts_d, patch_num)
        random_idx = torch.argsort(rand_tmp, dim = -1)
        masked_idxs = random_idx[:, :, :mask_num].sort(dim = -1)[0]
        unmasked_idxs = random_idx[:, :, mask_num:].sort(dim = -1)[0]

        return masked_idxs, unmasked_idxs