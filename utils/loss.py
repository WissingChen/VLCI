"""
Language Model loss
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output


def compute_im_loss(pred, imgs, mask, p=16, norm_pix_loss=False):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = patchify(imgs, p)
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    return loss


def patchify(imgs, p):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    # p = self.patch_embed.patch_size[0]
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x


def compute_lm_loss(output, reports_ids, reports_masks):
    if isinstance(output, list):
        output = output[0]
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss


def compute_recon_loss(pred, target, mask, mode='text'):
    if mode == 'text':
        loss = compute_lm_loss(pred, target, mask)
    elif mode == 'img':
        loss = compute_im_loss(pred, target, mask)
    else:
        raise ValueError
    return loss