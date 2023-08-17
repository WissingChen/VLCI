import torch
import torch.nn as nn
from torchvision.models import resnet101
from modules.modules4transformer import PositionalEncoding, Embeddings, LayerNorm, EncoderLayer, DecoderLayer, PositionwiseFeedForward, SublayerConnection, MultiHeadedAttention
from modules.pos_embed import get_2d_sincos_pos_embed
from utils import tensor_utils


class PatchEmbed(nn.Module):
    """
    resnet 1-3 block stem
    """

    def __init__(self, img_size=224, patch_size=16):
        super(PatchEmbed, self).__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        model = resnet101(True)
        modules = list(model.children())[:-3]
        self.embed = nn.Sequential(*modules)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.embed(x).flatten(2).transpose(1, 2)
        return x


class VisEmbed(nn.Module):
    """
    image embedding with 2d sin-cos position embedding
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=512, dropout=0.):
        super(VisEmbed, self).__init__()

        # --------------------------------------------------------------------------
        # SimVLM encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size)
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(1024, embed_dim)
        self.norm = LayerNorm(embed_dim)
        num_patches = self.patch_embed.num_patches
        # use 2d pos embed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        # self.norm = norm_layer(embed_dim)
        self.initialize_weights()

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.proj(x) + self.pos_embed[:, 1:, :]
        x = self.dropout(self.norm(x))
        return x


class TextEmbed(nn.Module):
    """
    test embedding with 1d sin-cos embedding
    """

    def __init__(self, embed_dim, vocab_size, dropout=0.):
        super(TextEmbed, self).__init__()
        self.embed = Embeddings(embed_dim, vocab_size)
        self.pos_encode = PositionalEncoding(embed_dim, dropout)
        self.norm = LayerNorm(embed_dim)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.pos_encode(self.embed(x)))
        return x


class MultiwayEncoderLayer(EncoderLayer):
    """
    weight-shared attention
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(MultiwayEncoderLayer, self).__init__(embed_dim, num_heads, ff_dim, dropout)
        self.feed_forward_t = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_ff_t = SublayerConnection(embed_dim, dropout)

    def forward(self, x, mask=None, hv_length=None, mode='m'):
        x = self.sublayer_attn(x, lambda x: self.attn(x, x, x, mask))

        if mode == 'v':
            x = self.sublayer_ff(x, self.feed_forward)
        elif mode == 't':
            x = self.sublayer_ff_t(x, self.feed_forward_t)
        else:
            if hv_length is None:
                raise ValueError("hv_length is None")
            x_v = x[:, :hv_length]
            x_l = x[:, hv_length:]
            x_v = self.sublayer_ff(x_v, self.feed_forward)
            x_l = self.sublayer_ff_t(torch.cat([x_v[:, :1], x_l], dim=1), self.feed_forward_t)
            x = torch.cat([x_v, x_l[:, 1:]], dim=1)
        return x


class MultiwayEncoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(MultiwayEncoder, self).__init__()
        self.layers = nn.ModuleList([MultiwayEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                     for _ in range(num_layer)])
        self.norm = LayerNorm(embed_dim)

    def forward(self, h, mask=None, hv_length=None, mode='m'):
        for layer in self.layers:
            h = layer(h, mask, hv_length, mode)
        h = self.norm(h)
        return h


class MultimodalDecoderLayer(DecoderLayer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super(MultimodalDecoderLayer, self).__init__(embed_dim, num_heads, ff_dim, dropout)
        self.cross_attn_v = MultiHeadedAttention(num_heads, embed_dim, dropout)
        self.feed_forward_v = PositionwiseFeedForward(embed_dim, ff_dim, dropout)

        self.sublayer_cross_v = SublayerConnection(embed_dim, dropout)
        self.sublayer_ff_v = SublayerConnection(embed_dim, dropout)

    def forward(self, x, h=None, self_mask=None, cross_mask=None, mode='plm'):
        x = self.sublayer_self(x, lambda x: self.self_attn(x, x, x, self_mask))

        if mode == 'plm':
            if h is not None:
                x = self.sublayer_cross(x, lambda x: self.cross_attn(x, h, h, cross_mask))
            x = self.sublayer_ff(x, self.feed_forward)
        else:
            if h is not None:
                x = self.sublayer_cross_v(x, lambda x: self.cross_attn_v(x, h, h, cross_mask))
            x = self.sublayer_ff_v(x, self.feed_forward_v)
        return x


class MultimodalDecoder(nn.Module):
    def __init__(self, embed_dim, num_layer, num_heads, ff_dim, dropout):
        super(MultimodalDecoder, self).__init__()
        # decoder layer
        self.layers = nn.ModuleList([MultimodalDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
                                        for _ in range(num_layer)])
        self.norm = LayerNorm(embed_dim)
        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward_mim(self, hv, ht=None, cross_mask=None, ids_restore=None, pos_embed=None):
        """ Masked Image Modeling """
        # embed tokens
        # hv = self.decoder_embed(hv)

        # append mask tokens to sequence
        cls_token = hv[:, :1, :]
        mask_tokens = self.mask_token.repeat(hv.shape[0], ids_restore.shape[1] + 1 - hv.shape[1], 1)
        hv = torch.cat([hv[:, 1:, :], mask_tokens], dim=1)  # no cls token
        hv = torch.gather(hv, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, hv.shape[2]))  # unshuffle
        hv = torch.cat([cls_token, hv], dim=1)  # append cls token

        # add pos embed
        hv = hv + pos_embed

        # apply Transformer blocks
        for layer in self.layers:
            hv = layer(hv, ht, None, cross_mask, mode='mim')

        hv = self.norm(hv)
        return hv

    def forward_plm(self, ht, hv=None, self_mask=None, cross_mask=None):
        """ Prefix Language Modeling """
        # if hv is not None:
        #     hv = self.decoder_proj(hv)
        for i in range(len(self.layers)):
            ht = self.layers[i](ht, hv, self_mask, cross_mask, mode='plm')
        ht = self.norm(ht)
        return ht


def get_hv_mask(hv):
    v_masks = hv.new_ones(hv.shape[:2], dtype=torch.long)
    v_masks = v_masks.unsqueeze(-2)
    return v_masks


def get_ht_mask(seq=None):
    if seq is not None:
        # crop the last one
        seq = seq[:, :-1]
        seq_mask = (seq.data > 0)
        seq_mask[:, 0] += True
        seq_mask = seq_mask.unsqueeze(-2)
        seq_mask = seq_mask & tensor_utils.subsequent_mask(seq.size(-1)).to(seq_mask)
    else:
        seq_mask = None
    return seq_mask, seq


def get_cross_mask(hv, seq):
    seq = seq[:, :-1]
    seq_mask = (seq.data > 0)
    seq_mask[:, 0] += True
    seq_mask = seq_mask.unsqueeze(-1)
    v_mask = hv.new_ones(hv.shape[:2], dtype=torch.long)
    v_mask = v_mask.unsqueeze(-2)

    cross_mask = seq_mask & v_mask
    return cross_mask
