"""
Visual Language Pre-training
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.modules4vlp import VisEmbed, TextEmbed, MultimodalDecoder, MultiwayEncoder
from utils import tensor_utils


class VLP(nn.Module):
    def __init__(self, args, tokenizer):
        super(VLP, self).__init__()
        self.args = args
        self.vocab_size = len(tokenizer.idx2token)
        self.tokenizer = tokenizer

        self.en_num_layers = args["en_num_layers"]
        self.de_num_layers = args["de_num_layers"]
        self.embed_dim = args["embed_dim"]
        self.num_heads = args["num_heads"]
        self.ff_dim = args["embed_dim"] * 4
        self.dropout = args["dropout"]

        # ------------------------------------------
        # Embedding
        self.vis_embed = VisEmbed(embed_dim=self.embed_dim, dropout=self.dropout)
        self.text_embed = TextEmbed(embed_dim=self.embed_dim, vocab_size=self.vocab_size + 1, dropout=self.dropout)

        # Encoder
        self.encoder = MultiwayEncoder(embed_dim=self.embed_dim, num_layer=self.en_num_layers,
                                       num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout)

        # Decoder
        self.decoder = MultimodalDecoder(embed_dim=self.embed_dim, num_layer=self.de_num_layers,
                                         num_heads=self.num_heads, ff_dim=self.ff_dim, dropout=self.dropout)

        self.logit = nn.Linear(self.embed_dim, self.vocab_size + 1)
        self.predict_patch = nn.Linear(self.embed_dim, 16 ** 2 * 3, bias=True)  # decoder to patch

        self.up = nn.Upsample(scale_factor=2)
        self.down = nn.MaxPool2d(3, 2, 1)
        # cls token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)

        if args["dataset_name"] == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args["dataset_name"] == 'ffa_ir':
            self.forward = self.forward_ffa_ir
        else:
            self.forward = self.forward_mimic_cxr

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_iu_xray(self, images, targets=None, mode='vt2t'):
        B = images.size(0)
        _image = torch.zeros_like(images[:, 0])
        # if mode == 'vt2t' or 'vt2v':
        for i in range(B):
            image_idx = 0
            if torch.rand(1) > 0.5:
                image_idx = 1
            _image[i] = images[i, image_idx]
        hv = self.vis_embed(self.up(self.down(_image)))  # SR
        # hv = self.vis_embed(_image)
        return self._forward(hv, _image, targets, mode)

    def forward_ffa_ir(self, images, targets=None, mode='img'):
        # todo
        hv = self.vis_embed(images)
        return hv, images

    def forward_mimic_cxr(self, images, targets=None, mode='img'):
        hv = self.vis_embed(self.up(self.down(images)))  # SR
        # hv = self.vis_embed(images)
        return self._forward(hv, images, targets, mode)

    def _forward(self, hv, image, targets, mode):

        # hv = self.decoder_proj(hv)
        hv = hv.reshape([image.size(0), -1, self.embed_dim])

        if mode == 'vt2t':
            """ all vis token & prefix text token, generate postfix text """
            # =====================
            # Preprocessing
            # =====================
            # append cls token
            cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(hv.shape[0], -1, -1)
            hv = torch.cat((cls_tokens, hv), dim=1)
            # split text and get mask
            prefix_text, postfix_text, idx = split_text(targets)
            prefix_mask = get_prefix_mask(hv, prefix_text)
            # o_mask, oh_mask, postfix_text = get_ht_mask(postfix_text, h_mask)
            self_mask = get_postfix_mask(None, postfix_text, None)
            cross_mask = get_postfix_mask(None, postfix_text, prefix_mask)
            # text embed
            postfix_text = postfix_text[:, :-1]
            ht_prefix = self.text_embed(prefix_text)  # [B, L] -> [B, L, D]
            ht_postfix = self.text_embed(postfix_text)  # [B, L] -> [B, L, D]
            # =====================
            # Modeling
            # =====================
            # encode
            h = torch.cat([hv, ht_prefix], dim=1)
            h = self.encoder(h, prefix_mask, hv_length=hv.size(1))
            # decode
            out = self.decoder.forward_plm(ht_postfix, h, self_mask=self_mask, cross_mask=cross_mask)
            outputs = F.log_softmax(self.logit(out), dim=-1)  # the vocab probs
            # marge
            mask = (postfix_text.data > 0).long()
            mask[:, 0] += True
            return outputs, postfix_text, mask

        elif mode == 'vt2v':
            """ vis keep token & all text token, generate vis remove token """
            # =====================
            # Preprocessing
            # =====================
            # masking: length -> length * mask_ratio
            hv_keep, mask, ids_restore, hv_remove = img_random_masking(hv, self.args["v_mask_ratio"],
                                                                       return_remove=True)
            # append cls token
            cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(hv_keep.shape[0], -1, -1)
            hv_keep = torch.cat((cls_tokens, hv_keep), dim=1)
            # get prefix mask
            targets = targets[:, :-1]
            prefix_mask = get_prefix_mask(hv_keep, targets)  # dual stream attn
            ht = self.text_embed(targets)  # [B, L] -> [B, L, D]
            cross_mask = get_postfix_mask(torch.cat((cls_tokens, hv), dim=1), None, get_prefix_mask(None, targets))
            # self_mask = get_postfix_mask(hv_remove, None, None)
            # =====================
            # Modeling
            # =====================
            # encode
            h = torch.cat([hv_keep, ht], dim=1)
            h = self.encoder(h, prefix_mask, hv_length=hv.size(1))
            # decode
            L = hv_keep.size(1)  # cls token + vis token
            hv_keep = h[:, :L]
            ht = h[:, L:]
            # outputs = self.decoder.forward_vis_v2(hv_remove, h, L, o_mask, oh_mask, ids_restore)
            outputs = self.decoder.forward_mim(hv_keep, ht, cross_mask,
                                               ids_restore=ids_restore, pos_embed=self.vis_embed.pos_embed)
            # predictor projection
            outputs = self.predict_patch(outputs)
            # remove cls token
            outputs = outputs[:, 1:, :]
            return outputs, image, mask

        elif mode == 'v2v':
            """ vis keep token, generate vis remove token """
            # =====================
            # Preprocessing
            # =====================
            # masking: length -> length * mask_ratio
            hv_keep, mask, ids_restore = img_random_masking(hv, self.args["v_mask_ratio"], return_remove=False)
            # append cls token
            cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(hv_keep.shape[0], -1, -1)
            hv_keep = torch.cat((cls_tokens, hv_keep), dim=1)
            # get prefix mask
            prefix_mask = get_prefix_mask(hv_keep, None)  # dual stream attn
            # =====================
            # Modeling
            # =====================
            # encode
            h = hv_keep
            h = self.encoder(h, prefix_mask, mode='v')
            # decode
            outputs = self.decoder.forward_mim(h, ids_restore=ids_restore, pos_embed=self.vis_embed.pos_embed)
            # predictor projection
            outputs = self.predict_patch(outputs)
            # remove cls token
            outputs = outputs[:, 1:, :]
            return outputs, image, mask

        elif mode == 't2t':
            """ prefix text token, generate postfix text """
            # =====================
            # Preprocessing
            # =====================
            # split text and get mask
            prefix_text, postfix_text, idx = split_text(targets)
            # append cls token
            cls_token = self.cls_token + self.vis_embed.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(prefix_text.shape[0], -1, -1)
            # text embed
            prefix_mask = get_prefix_mask(cls_tokens, prefix_text)
            ht_prefix = self.text_embed(prefix_text)  # [B, L] -> [B, L, D]
            ht_postfix = self.text_embed(postfix_text[:, :-1])  # [B, L] -> [B, L, D]
            ht_prefix = torch.cat([cls_tokens, ht_prefix], dim=1)
            self_mask = get_postfix_mask(None, postfix_text, None)
            cross_mask = get_postfix_mask(None, postfix_text, prefix_mask)

            # =====================
            # Modeling
            # =====================
            # encode
            h = ht_prefix
            h = self.encoder(h, prefix_mask, mode='t')
            # decode
            out = self.decoder.forward_plm(ht_postfix, h, self_mask=self_mask, cross_mask=cross_mask)
            outputs = F.log_softmax(self.logit(out), dim=-1)  # the vocab probs
            # marge
            postfix_text = postfix_text[:, :-1]
            mask = (postfix_text.data > 0).long()
            mask[:, 0] += True
            return outputs, postfix_text, mask

        else:
            raise ValueError


def img_random_masking(x, mask_ratio, return_remove=False):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device).long()
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    if return_remove:
        ids_remove = ids_shuffle[:, len_keep:]
        x_unmasked = torch.gather(x, dim=1, index=ids_remove.unsqueeze(-1).repeat(1, 1, D))
        return x_masked, mask, ids_restore, x_unmasked
    return x_masked, mask, ids_restore


def split_text(text):
    """
    split text into two subset a and b, a for encode and b for decode
    every set be like [bos, id1, id2, id3, id4, ...., eos]

    :param text: text id [bos, id1, id2, ..., eos, 0, 0, ...]
    :return: prefix, postfix, split_idx
    """
    N, L = text.shape  # batch, length, dim
    _L = (torch.sum(text > 0, dim=1) + 1).cpu()
    idx = (torch.rand(N) * (_L - 3) + 2).int()  # get Tp
    pre_l = idx.max() + 1
    past_l = (_L - idx).max() + 1
    pre_set = torch.zeros(N, pre_l).to(text.device)
    past_set = torch.zeros(N, past_l).to(text.device)
    # split
    for i in range(N):
        pre_set[i, :idx[i]] += text[i, :idx[i]]
        past_set[i, 1:_L[i] - idx[i] + 1] += text[i, idx[i]:_L[i]]

    return pre_set.long(), past_set.long(), idx


def get_prefix_mask(prefix_vis, prefix_text):
    """
    Dual stream attention for encoder

    :param prefix_vis: [B, L, N]
    :param prefix_text: [B, L], text seq
    """
    if prefix_text is not None:
        text_mask = (prefix_text.data > 0)
        text_mask[:, 0] += True
        if prefix_vis is not None:
            vis_mask = torch.ones(prefix_text.size(0), prefix_vis.size(1)).to(prefix_text) == 1
            prefix_mask = torch.cat([vis_mask, text_mask], dim=1)
        else:
            prefix_mask = text_mask
    else:
        if prefix_vis is None:
            raise ValueError
        else:
            prefix_mask = torch.ones(prefix_vis.size(0), prefix_vis.size(1)).to(prefix_vis) == 1

    prefix_mask = prefix_mask.unsqueeze(-2)
    prefix_mask = prefix_mask & (torch.ones(1, prefix_mask.size(-1), prefix_mask.size(-1)).to(prefix_mask) == 1)

    return prefix_mask


def get_ht_mask(seq, h_mask):
    # crop the last one
    seq = seq[:, :-1]
    seq_mask = (seq.data > 0)
    seq_mask[:, 0] += True
    oh_mask = seq_mask.unsqueeze(-1) & h_mask[:, :seq_mask.size(-1)]
    seq_mask = seq_mask.unsqueeze(-2)
    seq_mask = seq_mask & tensor_utils.subsequent_mask(seq.size(-1)).to(seq_mask)
    return seq_mask, oh_mask, seq


def get_postfix_mask(postfix_vis, postfix_text, prefix_mask):
    """
    postfix attention for decoder
    1. autoregression text self attention, input(postfix_text)
    2. autoregression text cross attention, input(postfix_text, prefix_mask)
    3. autoregression vis self attention, input(postfix_vis)
    4. autoregression vis cross attention, input(postfix_vis, prefix_mask)
    :param postfix_text: [B, L1]
    :param postfix_vis: [B, L2]
    :param prefix_mask: [B, L3, L3]
    :return: attention mask
    """
    if postfix_text is not None and postfix_vis is None:
        # crop the last one
        postfix_text = postfix_text[:, :-1]
        text_mask = (postfix_text.data > 0)
        text_mask[:, 0] += True
        if prefix_mask is None:
            # 1. autoregression text self attention
            postfix_mask = text_mask
            postfix_mask = postfix_mask.unsqueeze(-2)
            postfix_mask = postfix_mask & tensor_utils.subsequent_mask(postfix_text.size(-1)).to(postfix_mask)
        else:
            # 2. autoregression text cross attention
            postfix_mask = text_mask.unsqueeze(-1) & prefix_mask[:, :1].repeat(1, text_mask.size(1), 1)
    elif postfix_text is None and postfix_vis is not None:
        vis_mask = torch.ones(postfix_vis.size(0), postfix_vis.size(1)).to(postfix_vis) == 1
        if prefix_mask is None:
            # 3. autoregression vis self attention, input(postfix_vis)
            vis_mask = vis_mask.unsqueeze(-2)
            # dual stream
            postfix_mask = vis_mask & (torch.ones(1, vis_mask.size(-1), vis_mask.size(-1)).to(postfix_vis) == 1)
            # single stream
            # postfix_mask = vis_mask & tensor_utils.subsequent_mask(vis_mask.size(-1)).to(postfix_vis)
        else:
            # 4. autoregression vis cross attention, input(postfix_vis, prefix_mask)
            postfix_mask = vis_mask.unsqueeze(-1) & prefix_mask[:, :1].repeat(1, vis_mask.size(1), 1)
    else:
        raise ValueError
    return postfix_mask


def get_postfix_mask_old(text):
    """
    Dual stream attention
    hv_size
    pre_set [B, L]
    """
    seq_mask = (text.data > 0)
    seq_mask[:, 0] += True
    # hv_mask = torch.ones(text.size(0), 197).to(text.device)==1
    postfix_mask = seq_mask.unsqueeze(-2)
    postfix_mask = postfix_mask & (torch.ones(1, 197, postfix_mask.size(-1)).to(text.device) == 1)
    return postfix_mask
