import torch
import torch.nn as nn
from torchvision.models import resnet101
from modules.coatnet import Transformer as CoAtNetTrans, Rearrange
from modules.modules4transformer import PositionalEncoding, Embeddings, LayerNorm
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


class DownSamplingTrans(CoAtNetTrans):
    def __init__(self, **kwargs):
        super(DownSamplingTrans, self).__init__(**kwargs)
        self.vec2img = Rearrange('b (ih iw) c -> b c ih iw', ih=self.ih * 2, iw=self.iw * 2)
        self.img2vec = Rearrange('b c ih iw -> b (ih iw) c', ih=self.ih, iw=self.iw)

    def forward(self, x):
        x = self.vec2img(x)
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(x))
        else:
            x = x + self.attn(x)
        x = x + self.ff(x)
        return self.img2vec(x)


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
