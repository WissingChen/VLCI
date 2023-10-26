import json
import random
from collections import OrderedDict

import numpy as np
import torch
import argparse
from metric.metrics import compute_scores
from utils import R2DataLoader, tokenizers_fn, build_optimizer, build_lr_scheduler, loss_fn
from trainer import *
import os
from models import model_fn
import warnings

warnings.filterwarnings('ignore')


def load_json_args(path):
    json_str = ''
    with open(path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'  #
            json_str += line
    defaults = json.loads(json_str, object_pairs_hook=OrderedDict)
    dict_args = {}
    for key in defaults.keys():
        dict_args.update(defaults[key])
    return dict_args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # -------------------------------
    # load hyper-param
    # -------------------------------
    parse = argparse.ArgumentParser()
    parse.add_argument('--c', type=str, default='config/iu_xray/vlci.json',
                       help='json file of config')
    json_path = parse.parse_args()
    args = load_json_args(json_path.c)
    torch.cuda.set_device(int(args["cuda"]))
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # -------------------------------
    # fix random seeds
    # -------------------------------
    if args["seed"] == -1:
        args["seed"] = np.random.randint(0, 23333)
    print(args)
    setup_seed(args["seed"])
    # -------------------------------
    # create tokenizer
    # -------------------------------
    tokenizer = tokenizers_fn[args['tokenizer']](args)
    print('count of tokens', len(tokenizer.token2idx))
    # -------------------------------
    # create data loader
    # -------------------------------
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    # -------------------------------
    # build model architecture
    # -------------------------------
    model = model_fn[args["model"]](args, tokenizer)
    model = model.cuda()
    # -------------------------------
    # get function handles of loss and metrics
    # -------------------------------
    criterion = loss_fn[args["loss_fn"]]
    metrics = compute_scores
    # -------------------------------
    # build optimizer, learning rate scheduler
    # -------------------------------
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer, len(train_dataloader))

    # -------------------------------
    # build trainer and start to train
    # -------------------------------
    kwarg = {"model": model, "criterion": criterion, "metric_ftns": metrics, "optimizer": optimizer, "args": args,
             "lr_scheduler": lr_scheduler, "train_dataloader": train_dataloader, "val_dataloader": val_dataloader,
             "test_dataloader": test_dataloader}

    if args["task"] == 'finetune':
        trainer = FTrainer(**kwarg)
        trainer.train()
    elif args["task"] == 'pretrain':
        trainer = PTrainer(**kwarg)
        trainer.train()
    else:
        trainer = Trainer(**kwarg)
        trainer.inference()



if __name__ == '__main__':
    main()
