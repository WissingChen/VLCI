"""
optimizers and lr scheduler
"""
import torch
import math


def build_optimizer(args, model):
    task = args["task"]
    optim = args["optim"]
    weight_decay = args["weight_decay"]
    lr_en = args["lr_en"]
    lr_de = args["lr_de"]
    amsgrad = args["amsgrad"]
    if task == 'train':
        en_params = list(map(id, model.vis_embed.parameters()))
        de_params = filter(lambda x: id(x) not in en_params, model.parameters())
        optimizer = getattr(torch.optim, optim)(
            [{'params': model.vis_embed.parameters(), 'lr': lr_en, "lr_scale": 1.},
             {'params': de_params, 'lr': lr_de}],
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    else:
        optimizer = getattr(torch.optim, optim)(
            [{'params': model.parameters(), 'lr': lr_de}],
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    return optimizer


def build_lr_scheduler(args, optimizer, l):
    lr_scheduler_fn = args["lr_scheduler"]
    step_size = args["step_size"]
    epochs = args["epochs"]
    if lr_scheduler_fn == 'warmup_steplr':
        lr_scheduler = WarmupAndSteplr(optimizer, step_size * l, epochs * l)
    elif lr_scheduler_fn == 'warmup':
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, step_size * l, epochs * l)
    else:
        lr_scheduler = getattr(torch.optim.lr_scheduler, args["lr_scheduler"])(optimizer, step_size, args["gamma"])
    print(f"Build {lr_scheduler_fn} for {args['optim']} in {args['task']}")
    return lr_scheduler


class WarmupAndSteplr(object):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
        self.end_lr = 0.001
        self.current_step = 0
        self.optim = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.lr_ve = self.optim.param_groups[0]['lr']
        self.lr_ed = self.optim.param_groups[-1]['lr']

    def step(self):
        self.current_step += 1
        if self.current_step < self.num_warmup_steps:
            lr_1 = float(self.current_step) / float(max(1, self.num_warmup_steps))
        else:
            # cosine annealing decay
            progress = float(self.current_step - self.num_warmup_steps) / float(
                max(1, self.num_training_steps - self.num_warmup_steps))
            cosine_lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
            # lr = max(0.0, cosine_lr * (base_lr - end_lr) + end_lr)
            lr_1 = max(0.0, cosine_lr * (1 - self.end_lr) + self.end_lr)

        # if self.current_step < self.num_warmup_steps:
        #     lr_2 = lr_1
        if self.current_step == self.num_training_steps // 2:
            lr_2 = .1
        else:
            lr_2 = 1.

        for param_group in self.optim.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr_1 * param_group["lr_scale"] * self.lr_ve
            else:
                param_group["lr"] = lr_2 * self.lr_ed
        return lr_1 * self.lr_ve, lr_2 * self.lr_ed


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    when current_step < num_warmup_steps，
    new_lr =current_step/num_warmup_steps * base_lr
    when current_step >= num_warmup_steps，
    new_lr =(num_training_steps - current_step) / (num_training_steps -num_warmup_steps) * base_lr

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_line(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    def lr_cosine(current_step: int):
        # linear warmup
        end_lr = 0.001
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # cosine annealing decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_lr = max(0.0, 0.5 * (1. + math.cos(math.pi * progress)))
        # lr = max(0.0, cosine_lr * (base_lr - end_lr) + end_lr)
        lr = max(0.0, cosine_lr * (1 - end_lr) + end_lr)
        return lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_cosine, last_epoch)


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75, base_lr=1e-3):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    try:
        num_layers = len(model.blocks) + 1
    except:
        num_layers = len(model.layers) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_vit(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": base_lr,
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
                "lr": base_lr,
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('layers'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers
