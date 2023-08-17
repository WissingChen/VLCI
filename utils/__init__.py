from .loss import compute_lm_loss, compute_recon_loss
from .dataloaders import R2DataLoader
from .tokenizers import Tokenizer, MixTokenizer
from .optimizers import build_optimizer, build_lr_scheduler

loss_fn = {'lm': compute_lm_loss, 'recon': compute_recon_loss}
tokenizers_fn = {'ori': Tokenizer, 'mix': MixTokenizer}
