from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange, repeat

class encoder(nn.Module):
    def __init__(self, n_layers=3, d_model=256, n_heads=4,  d_ff=512, dropout=0.):
        super(encoder, self).__init__()
        #the encoder does not handle the patch staff, just encoded the given patches
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first = True)
        self.encoder_layers = TransformerEncoder(encoder_layer, n_layers)
    
    def forward_uni(self, patch, mask = None):
        batch_size, patch_num, _ = patch.shape

        encoded_patch = self.encoder_layers(patch, src_key_padding_mask = mask)

        return encoded_patch
            
    def forward_multi(self, patch):
        batch_size, ts_d, patch_num, _ = patch.shape

        patch_mergeD = rearrange(patch, 'b ts_d patch_num d_model -> (b ts_d) patch_num d_model')
        encoded_patch_mergeD = self.encoder_layers(patch_mergeD)
        encoded_patch = rearrange(encoded_patch_mergeD, '(b ts_d) patch_num d_model -> b ts_d patch_num d_model', b = batch_size)

        return encoded_patch

class decoder(nn.Module):
    def __init__(self, patch_size, n_layers=1, d_model=256, n_heads=4,  d_ff=512, dropout=0.):
        super(decoder, self).__init__()
        #the decoder takes encoded tokens + indicating tokens as input, projects tokens to original space
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        decoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first = True)
        self.decoder_layers = TransformerEncoder(decoder_layer, n_layers)

        self.output_layer = nn.Linear(d_model, patch_size)
    
    def forward_uni(self, patch):
        batch_size, patch_num, _ = patch.shape

        decoded_patch = self.decoder_layers(patch)
        decoded_ts = self.output_layer(decoded_patch)
        decoded_ts = rearrange(decoded_ts, 'b patch_num patch_size -> b (patch_num patch_size)')

        return decoded_ts
            
    def forward_multi(self, patch):
        batch_size, ts_d, patch_num, _ = patch.shape

        patch_mergeD = rearrange(patch, 'b ts_d patch_num d_model -> (b ts_d) patch_num d_model')
        decoded_patch_mergeD = self.decoder_layers(patch_mergeD)
        decoded_patch = rearrange(decoded_patch_mergeD, '(b ts_d) patch_num d_model -> b ts_d patch_num d_model', b = batch_size)
        decoded_ts = self.output_layer(decoded_patch)
        decoded_ts = rearrange(decoded_ts, 'b ts_d patch_num patch_size -> b ts_d (patch_num patch_size)')

        return decoded_ts