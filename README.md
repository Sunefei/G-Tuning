# Fine-tuning Graph Neural Networks by Preserving Graph Generative Patterns
 <a href='https://aaai.org/aaai-conference/aaai-24-photos/'><img src='https://img.shields.io/badge/Conference-AAAI-magenta'></a> 
 <a href='https://github.com/horrible-dong/DNRT/blob/main/LICENSE'><img src='https://img.shields.io/badge/License-Apache--2.0-blue'></a> 
## G-Tuning
This repo is the implementation of paper “Fine-tuning Graph Neural Networks by Preserving Graph Generative Patterns" accepted by AAAI' 24.

## Full paper with appendix
Paper on arXiv [Click Here](https://arxiv.org/abs/2312.13583)


## Citation

```latex
@inproceedings{sun2024fine,
  title={Fine-Tuning Graph Neural Networks by Preserving Graph Generative Patterns},
  author={Sun, Yifei and Zhu, Qi and Yang, Yang and Wang, Chunping and Fan, Tianyu and Zhu, Jiajun and Chen, Lei},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={8},
  pages={9053--9061},
  year={2024}
}
```


## Motivation
We identify the generative patterns of downstream graphs as a crucial aspect in narrowing the gap between pre-training and fine-tuning.

![IntroFig](./figs/intro_all_modified.png)

## Architecture
Building upon on our theoretical results, we design the model architecture **G-Tuning** to efficiently reconstruct graphon as generative patterns with rigorous generalization results.

![ModelFig](./figs/model.png)


## Environment Set-up

Please first clone the repo and install the required environment, which can be done by running the following commands:

The script has been tested running under Python 3.7.10, with the following packages installed (along with their dependencies):

- PyTorch. Version >=1.4 required. You can find instructions to install from source here.

- DGL. 0.5 > Version >=0.4.3 required. You can find instructions to install from source here.

- rdkit. Version = 2019.09.2 required. It can be easily installed with conda install -c conda-forge rdkit=2019.09.2

Other Python modules. Some other Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

```bash
git clone https://github.com/zjunet/G-Tuning.git
cd G-Tuning/
pip install -r requirements.txt
```
> [!NOTE]
> In addition, CUDA 10.0 has been used in our project. Although not all dependencies are mentioned in the installation instruction links above, you can find most of the libraries in the package repository of a regular Linux distribution.


## Run
Because our method is a **plug-and-play** graph tuning method, we can directly use any graph self-supervised method to obtain a pretrained model, or use any already graph trained model, such as:
- [GCC](https://github.com/THUDM/GCC):
```bash
python scripts/download.py --url https://drive.google.com/open?id=1lYW_idy9PwSdPEC7j9IH5I5Hc7Qv-22- --path saved --fname pretrained.tar.gz
# For regions where Google is not accessible, use
# python scripts/download.py --url https://cloud.tsinghua.edu.cn/f/cabec37002a9446d9b20/?dl=1 --path saved --fname pretrained.tar.gz
```
- [pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns.git):
```bash
wget https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model_gin/contextpred.pth
```

Alternatively, you can use your own pretrained model and your own pretraining method to obtain a graph pretrained model, but you will need to modify the model architecture and corresponding interfaces during the tuning stage.

Then, we use our improved tuning method to tune any existing graph pretrained model.
```bash
python train_G_Tuning.py --resume <pre-trained model file> \
  --dataset <downstream dataset>
  --model-path <fine-tuned model saved file> \
  --gpu <gpu id> \
  --epochs <epoch number> \
  --finetune
```
For more detail, the help information of the main script `train_G_Tuning.py`:

```bash
optional arguments:
  --epochs EPOCHS       number of training epochs (default:30)
  --optimizer {sgd,adam,adagrad}
                        optimizer (default:adam)
  --learning_rate LEARNING_RATE  learning rate (default:0.005)
  --resume PATH        path for pre-trained model (default: GCC)
  --dataset {the dataset name str}
  --hidden-size HIDDEN_SIZE  (default:64)
  --model-path MODEL_PATH    path to save fine-tuned model (default:saved)
  --finetune            whether to conduct fine-tune
  --gpu GPU [GPU ...]   GPU id to use.
```

## Contact
If you have any question about the code or the paper, feel free to contact me through [email](mailto:yifeisun@zju.edu.cn).

## Acknowledgements
Part of this code is inspired by Qiu et al.'s [GCC: Graph Contrastive Coding](https://github.com/THUDM/GCC).


