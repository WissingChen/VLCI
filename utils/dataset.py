import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args["image_dir"]
        self.ann_path = args["ann_path"]
        self.max_seq_length = args["max_seq_length"]
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        if args["dataset_name"] == 'ffa_ir': self.dict2list4ffair()
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

    def dict2list4ffair(self):
        examples_list = []
        for k, v in self.examples.items():
            v['id'] = k
            v['image_path'] = v.pop('Image_path')
            v['report'] = v.pop('En_Report')
            examples_list.append(v)
        self.examples = examples_list


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MixSingleImageDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = {'iu_xray': '../../MRG/data/iu_xray/images/',
                          'mimic_cxr': '../../MRG/data/mimic_cxr/images/'}
        self.ann_path = {'iu_xray': "../../MRG/data/iu_xray/annotation.json",
                         'mimic_cxr': "../../MRG/data/mimic_cxr/annotation.json"}
        self.max_seq_length = args["max_seq_length"]
        self.split = split
        self.tokenizer = tokenizer  # vocab + <unk>
        self.transform = transform
        self.ann = {'iu_xray': json.loads(open(self.ann_path['iu_xray'], 'r').read()),
                    'mimic_cxr': json.loads(open(self.ann_path['mimic_cxr'], 'r').read())}

        self.examples = self.ann['iu_xray'][self.split]
        length = len(self.examples)
        for i in range(length):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'], dataset='iu_xray')[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

        self.examples += self.ann['mimic_cxr'][self.split]
        for i in range(length, len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'], dataset='mimic_cxr')[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        if len(image_path) > 1:
            if torch.rand(1) > 0.5:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[0])).convert('RGB')
            else:
                image = Image.open(os.path.join(self.image_dir['iu_xray'], image_path[1])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        else:
            image = Image.open(os.path.join(self.image_dir['mimic_cxr'], image_path[0])).convert('RGB')
            if self.transform is not None:
                image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
