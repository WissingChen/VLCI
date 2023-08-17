"""
Finetune trainer for report generation
"""
import pandas as pd
from .BaseTrainer import BaseTrainer
import numpy as np
import torch
import time
from trainer.PretrainTrainer import unpatchify, vis_heatmap
from utils.loss import patchify
import os


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args)
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.best_score = 0.

        pretrained_dict = torch.load(args["load_model_path"], map_location='cuda')['state_dict']
        # if args["task_name"].split('_')[1] == 'en':
        #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if
        #                        k.split('.')[0] != 'decoder' and k.split('.')[0] != 'logit'}
        # print(pretrained_dict.keys())
        self.model.load_state_dict(pretrained_dict, False)


    def _train_epoch(self, epoch):

        train_loss = 0
        self.model.train()
        start_time = time.time()
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.train_dataloader):
            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            output = self.model(images, reports_ids, mode='train')
            nll_loss = self.criterion(output, reports_ids, reports_masks)
            loss = nll_loss
            self.optimizer.zero_grad()
            loss.backward()
            train_loss += loss.item()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            print(f"\repoch: {epoch} {batch_idx}/{len(self.train_dataloader)}\tloss: {loss:.3f}\tmean loss: {train_loss/(batch_idx+1):.3f}",
                  flush=True, end='')

            if self.args["lr_scheduler"] != 'StepLR':
                self.lr_scheduler.step()
        if self.args["lr_scheduler"] == 'StepLR':
            self.lr_scheduler.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print("\n")
        print("\tEpoch {}\tmean_loss: {:.4f}\ttime: {:.4f}s".format(epoch, log['train_loss'], time.time() - start_time))

        self.model.eval()
        with torch.no_grad():
            val_gts, val_res = [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.val_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                val_res.extend(reports)
                val_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rVal Processing: [{int((batch_idx + 1) / len(self.val_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            # record val metrics
            for k, v in val_met.items():
                self.monitor.logkv(key='val_' + k, val=v)
            val_met['p'] = lp
            log.update(**{'val_' + k: v for k, v in val_met.items()})

        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p = [], [], []
            p = torch.zeros([1, self.args["max_seq_length"]]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                p = torch.cat([p, output])
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            for k, v in test_met.items():
                self.monitor.logkv(key='test_' + k, val=v)
            test_met['p'] = lp
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        if self.args['monitor_metric_curves']:
            self.monitor.plot_current_metrics(epoch, self.monitor.name2val)
        self.monitor.dumpkv(epoch)
        return log

    def test_epoch(self):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p, img_ids = [], [], [], []
            hvs, hvs_id = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                # print(output[0])
                # print(reports[0])
                # marked_img = mark_local(images, part_inx)
                # vis_img(marked_img[0], title='high attn')
                # vis_img(torch.cat([images[0, 0], images[0, 1]], dim=-1), title='ori')

                ground_truths = self.model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            # test_res = torch.load("results/mimic_cxr/DMIRG/DMIRG/118_report_100.npy")
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            
            save_report(test_res, test_gts, img_ids, os.path.join(self.checkpoint_dir, 'report.csv'))
            print(test_met)
            # print(lp)

    def local_feature(self, idx):
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res, p, img_ids = [], [], [], []
            hvs, hvs_id = [], []
            p = torch.zeros([1, self.args.max_seq_length]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                if idx in images_id:
                    _, m = self.model(images, mode='extract')
                    attn = m['attn']

                    encoder_attn = encoder_heatmap(images, attn)
                    for i in range(encoder_attn.size(0)):
                        if idx == images_id[i]:
                            vis_heatmap(images[i], encoder_attn[i], title='encoder_attn')
                            print(images_id['i'])

    def keyword_feature(self, idx):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                if idx in images_id:
                    key_id = self.model.tokenizer.token2idx['pleural']
                    target = torch.tensor([0, key_id, 0]).to(images).long()
                    target = target.unsqueeze(0).repeat(images.size(0), 1)
                    attn = self.model(images, target, mode='keyword')

                    encoder_attn = decoder_heatmap(images, attn)
                    for i in range(encoder_attn.size(0)):
                        if idx == images_id[i]:
                            vis_heatmap(images[i], encoder_attn[i], title='encoder_attn')
                            print(images_id[i])

    def extract(self):
        self.model.eval()
        with torch.no_grad():
            hvs, hvs_id = [], []
            p = torch.zeros([1, self.args.max_seq_length]).cuda()
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(self.test_dataloader):
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                h, _ = self.model(images, mode='extract')
                _hvs, _hvs_id = self.select_feature(h, images_id)
                hvs.extend(_hvs)
                hvs_id.extend(_hvs_id)
                print(f"\rTest Processing: [{int((batch_idx + 1) / len(self.test_dataloader) * 100)}%]", end='',
                      flush=True)
            tp, lp = count_p(p[1:])
            # hvs_id = hvs_id.
            torch.save({'id': hvs_id, 'feature': hvs}, 'results/mimic_cxr/DMIRG/DMIRG/118_feature.npy')

    def select_feature(self, h, images_id):
        hv_list = []
        hv_ids = []
        h = h.detach().cpu().numpy()
        for i in range(h.shape[0]):
            if images_id[i] in self.for_tsne:
                hv_list.append(h[i])
                hv_ids.append(images_id[i])
        return hv_list, hv_ids


def count_p(p):
    t = torch.unique(p, dim=0)
    l = t.size(0)
    return t, l


def mark_local(img, inx):
    """
    img [B, [N], 3, H, W]
    output [B, 3, H, W]
    """
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        for i in range(B):
            mask[i, inx[i, :]] = 1

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)

    else:
        B, C, H, W = img.size()
        img = patchify(img, 16)
        mask = torch.zeros([B, 14 * 14]).to(img.device)
        for i in range(B):
            mask[i, inx[i, :]] = 1
        output = unpatchify(img, mask)

    return output


def encoder_heatmap(img, attn):
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)

    else:
        # B, C, H, W = img.size()
        mask = torch.zeros_like(img).to(img)
        mask = patchify(mask, 16)  # B, L, N

        length = len(attn)
        last_map = attn[0]
        for i in range(1, length):
            last_map = torch.matmul(attn[i], last_map)
        last_map = last_map[:, :, 0, 1:]
        attn_map = last_map.mean(dim=1).unsqueeze(-1)
        mask += attn_map
        output = unpatchify(mask)

    return output


def decoder_heatmap(img, attn):
    if len(img.size()) > 4:
        B, N, C, H, W = img.size()
        img1 = patchify(img[:, 0], 16)
        img2 = patchify(img[:, 1], 16)
        mask = torch.zeros([B, 14 * 14 * 2]).to(img.device)
        res_img = torch.ones_like(img1).to(img.device)

        res_img1 = unpatchify(res_img, 1 - mask[:, :14 * 14])
        res_img2 = unpatchify(res_img, 1 - mask[:, 14 * 14:])
        img1 = unpatchify(img1, mask[:, :14 * 14]) + res_img1
        img2 = unpatchify(img2, mask[:, 14 * 14:]) + res_img2
        output = torch.cat([img1, img2], dim=-1)

    else:
        attn = attn[0] + attn[1] + attn[2]
        # B, C, H, W = img.size()
        mask = torch.zeros_like(img).to(img)
        mask = patchify(mask, 16)  # B, L, N

        last_map = attn[:, :, 1, 1:]
        attn_map = last_map.mean(dim=1).unsqueeze(-1)
        mask += attn_map
        output = unpatchify(mask)

    return output


def save_report(inference, reference, ids, output_dir):
    df = pd.DataFrame({'Report Impression': inference})
    df.to_csv(output_dir)
