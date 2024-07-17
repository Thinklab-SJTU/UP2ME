# region
import torch
import torch.nn as nn
import sys
sys.path.append('.')
from einops import rearrange, repeat
from loguru import logger
# endregion


def generate_mask(ts, min_mask_ratio, max_mask_ratio):
    '''
    ts: [batch_sizce, ts_d, ts_len] 
    '''

    batch_size, ts_d, ts_len = ts.shape
    mask = torch.zeros_like(ts, dtype=int)

    min_mask_len = int(min_mask_ratio * ts_len)
    max_mask_len = int(max_mask_ratio * ts_len)
    mask_len = torch.randint(low=min_mask_len, high=max_mask_len + 1, size=(batch_size, ts_d), dtype=torch.int64)
    highs = ts_len - mask_len + 1
    lows = torch.zeros_like(highs)
    start = torch.cat([torch.randint(low, high, (1,)) for low, high in zip(lows.flatten(), highs.flatten())])
    start = start.reshape(batch_size, ts_d)

    mask = torch.arange(ts_len).unsqueeze(0).unsqueeze(1).repeat(batch_size, ts_d, 1) >= start.unsqueeze(-1)
    mask &= torch.arange(ts_len).unsqueeze(0).unsqueeze(1).repeat(batch_size, ts_d, 1) < (start + mask_len).unsqueeze(-1)
    mask = mask.to(ts.device)

    ts_masked = ts.masked_fill(mask == 1, 0)
    return ts_masked, mask, start, mask_len  # ! start / mask_len: batch_size* ts_d


if __name__ == '__main__':
    ts = torch.randn([1, 6, 720])
    masked_ts, mask = generate_mask(ts, 0.125, 0.25)
    print(masked_ts.shape)
    print(mask.shape)
