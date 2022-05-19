# FIERCE-repo
This repo contains the code for FIERCE, as presented in our paper "Preserving Fine-Grain Feature Information in
Classification via Entropic Regularization". We also provide the hyperspectral dataset that we used in Section 5.2. Note that we do not provide the dataset age estimation (IMDB & WIKi) and CUBs.

## Requirements and Installation:
1. A Pytorch installation https://pytorch.org
2. Python version 3.8.1 (lower versions might not work)
3 (optional). Age estimation dataset https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/: IMDB (faces only) and WIKI (faces only)
4 (optional). Cub dataset: https://deepai.org/dataset/cub-200-2011

## Datasets
Expect for Age estimation and Cub, the code will download automatically dataset of CIFARFS.

### Age Estimation:
it is required to run the following commands to pre-process the dataset (https://github.com/SITE5039/AdaMixUp)
$ python datasets/process_data_wiki_imdb.py --rootpath 'path/imdb_crop' --metafile 'imdb' --outfile  'imdbfilelist.txt'
$ python datasets/process_data_wiki_imdb.py --rootpath 'path/data/wiki_crop' --metafile 'wiki' --outfile  'wikifilelist.txt'

### CUBS
We use the split recommanded in Hu, Y., Gripon, V., & Pateux, S. (2020). Exploiting unsupervised inputs for accurate few-shot classification.

## Experiments

### Hyperspectral (folder hyperspectral)
- Experiments and ablation studies can be carried out with the notebooks:

### Age regression (folder age_regression)
training and saving results:
$ python main.py --alpha 0 --step 100 (Vanilla)
$ python main.py --alpha 0.2 --step 100 (FIERCE with parameter 0.3)
$ python ls.py (Label Smoothing)

models:
- We provide also 3 saved models: model_vanilla.pt (Cross-Entropy Criterion), model_fierce_0.2.pt (FIERCE with parameter 0.2), model_ls.pt (label smoothing)
plotting results & statistics:
- use notebook:


### Few Shot Experiments
training Cifas and print statistics
$ python main.py --dataset cifarfs --model resnet18 --lr 0.1 --skip-epochs 300 --entropy 0 --runs 10 (Vanilla)
$ python main.py --dataset cifarfs --model resnet18 --lr 0.1 --skip-epochs 300 --entropy 1.5 runs 10 (FIERE with parameter 1.5)
$ python main.py --dataset cifarfs --model resnet18 --lr 0.1 --skip-epochs 300 --entropy 0 --label-smothing 0.1 runs 10 (Label Smoothing with parameter $\sigma$ = 0.1)

training Cubfs and print statistics
$ python main.py --dataset cubfs --runs 10 --entropy-parameter 0 (Vanilla)
$ python main.py --dataset cubfs --runs 10 --entropy-parameter 2 (FIERCE with parameter 2)
$ python main.py --dataset cubfs --runs 10 --entropy-parameter 0 --label-smoothing 0.1  (Label Smoothing with parameter 0.1)

