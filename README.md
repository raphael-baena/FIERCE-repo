# FIERCE-repo
This repo contains the code for FIERCE, as presented in our paper *Preserving Fine-Grain Feature Information in
Classification via Entropic Regularizatio* (https://arxiv.org/abs/2208.03684). We also provide the hyperspectral dataset that we used in Section 5.2. Note that we do not provide the dataset age estimation (IMDB & WIKi) and CUB.

## Requirements and Installation:
1. A Pytorch installation https://pytorch.org
2. Python version 3.8.1 (lower versions might not work)
3. (optional) Age estimation dataset https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/: IMDB (faces only) and WIKI (faces only)
4. (optional) CUB dataset: https://deepai.org/dataset/cub-200-2011

## Datasets
Age estimation and CUB requires to download manually the datasets.

### Age Estimation:
it is required to run the following commands to pre-process the dataset.
```
$ python datasets/process_data_wiki_imdb.py --rootpath 'path/imdb_crop' --metafile 'imdb' --outfile  'imdbfilelist.txt'
$ python datasets/process_data_wiki_imdb.py --rootpath 'path/data/wiki_crop' --metafile 'wiki' --outfile  'wikifilelist.txt'
```
### CUB
We use the split recommanded in https://github.com/icoz69/DeepEMD.

## Experiments

### Hyperspectral (folder **hyperspectral**)
- Experiments and ablation studies can be carried out with the notebook: **Experiments_Hyperspectral.ipnyb**

### Age regression (folder **age_regression**)
- training and saving results:
Vanilla
```
$ python main.py 0  100
```
FIERCE with parameter 0.3
```
$ python main.py 0.2 100
```
Label Smoothing
```
$ python ls.py 0 100
```
- plotting the results: with the notebook *Stats_age_regression.ipynb*. It is possible to plot the results (MSE, prediction) from the file *features.pt* generated during the training of the model. We provide the features of the 3 models displayed in our paper (Cross-Entropy, Label Smoothing, FIERCE).
- computing mutual information, Stability and Reliability Diagram:
```
$ python stats.py pathtomodel
```
models:
- We provide also 3 saved models: model_vanilla.pt (Cross-Entropy Criterion), model_fierce_0.2.pt (FIERCE with parameter 0.2), model_ls.pt (label smoothing)
plotting results & statistics:


### Few Shot Experiments
- training CIFARFS and print statistics
Vanilla
```
$ python main.py --dataset cifarfs --model resnet18 --lr 0.1 --skip-epochs 300 --entropy 0 --runs 10
```
FIERCE with parameter 1.5
```
$ python main.py --dataset cifarfs --model resnet18 --lr 0.1 --skip-epochs 300 --entropy 1.5 runs 10
```
Label Smoothing
```
$ python main.py --dataset cifarfs --model resnet18 --lr 0.1 --skip-epochs 300 --entropy 0 --label-smothing 0.1 runs 10 
```
- training CUB and print statistics
```
$ python main.py --dataset cubfs --runs 10 --entropy-parameter 0 
$ python main.py --dataset cubfs --runs 10 --entropy-parameter 2 
$ python main.py --dataset cubfs --runs 10 --entropy-parameter 0 --label-smoothing 0.1 
```
