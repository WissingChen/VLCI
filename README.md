# VLCI

This is the implementation of [Visual-Linguistic Causal Intervention for Radiology Report Generation](https://arxiv.org/pdf/2303.09117.pdf).
It contains the codes of inference on IU-Xray/MIMIC-CXR dataset, and __we will release the codes of training soon__.


## Requirements
- `python==3.7`
- `numpy`
- `pytorch`
- `timm`
- `visdom`

## Preparation
1. Datasets: 
You can download the dataset via `data/datadownloader.py`, or download from the repo of [R2Gen](https://github.com/cuhksz-nlp/R2Gen).
Then, unzip the files into `data/iu_xray` and `data/mimic_cxr`, respectively. 
2. Models: We provide the well-trained models of VLCI for inference, and you can download from [here](https://1drv.ms/f/s!Ap3RrxWNCbeRpVKJPwM10D3hi4uE?e=O9Gtg3).
3. Please remember to change the path of data and models in the config file (`config/*.json`).

## Evaluation
- For VLCI on IU-Xray dataset 

`python main.py -c config/iu_xray/vlci.json`

The results of BLEU-4 metric will be: Val: 0.1661426319654664 Test: 0.1825833507943773

- For VLCI on MIMIC-CXR dataset

`python main.py -c config/mimic_cxr/vlci.json`

The results of BLEU-4 metric will be: Val: 0.14546148694468147 Test: 0.12331782668806458

## Citation
If you use this code for your research, please cite our paper.

```
@article{chen2023visual,
  title={Visual-Linguistic Causal Intervention for Radiology Report Generation},
  author={Chen, Weixing and Liu, Yang and Wang, Ce and Li, Guanbin and Zhu, Jiarui and Lin, Liang},
  journal={arXiv preprint arXiv:2303.09117},
  year={2023}
}
```

If you have any question about this code, feel free to reach me (chen867820261@gmail.com)