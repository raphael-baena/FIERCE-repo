import random
import torch
import numpy as np
import scipy.stats as st
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import time
import argparse
import sys
from datasets import MyDatasets
from torchvision import transforms
import matplotlib.pyplot as plt
data_set_path = "/users/local"

path = str(sys.argv[1])
num_classes = 2
device = torch.device("cpu")
criterion = nn.CrossEntropyLoss()
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, feature_maps, num_classes=num_classes):
        super(ResNet, self).__init__()
        self.in_planes = feature_maps

        self.length = len(num_blocks)
        self.conv1 = nn.Conv2d(3, feature_maps, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        layers = []
        for i, nb in enumerate(num_blocks):
            layers.append(self._make_layer(block, (2 ** i) * feature_maps, nb, stride = 1 if i == 0 else 2))            
            self.layers = nn.Sequential(*layers)
#        self.embedding = nn.Linear((2 ** (len(num_blocks) - 1)) * feature_maps * block.expansion, 1) 
        self.linear = nn.Linear(1, num_classes)
        self.depth = len(num_blocks)
        self.bounds = nn.Parameter(torch.Tensor([0.,1.]))
#        self.bins = nn.Parameter(torch.linspace(1, steps, steps))
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i in range(len(strides)):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            if i < len(strides) - 1:
                layers.append(nn.ReLU())
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        for i in range(len(self.layers)):
            out = self.layers[i](out)
        out = F.avg_pool2d(out, out.shape[2])
        features = out.view(out.size(0), -1)
#        features = self.embedding(features) #self.linear(features)
        features = features.mean(dim = 1, keepdim = True)
        features = self.bounds[1] * (features - self.bounds[0])
        #out = self.linear(features)
        return features

model = ResNet(BasicBlock, [2, 2, 2, 2], 16)
model.load_state_dict(  torch.load(path,map_location=torch.device('cpu')))

print(model)
transformations = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = MyDatasets.IMDBWIKI('data_set_path/imdb_crop', 'data_set_path/imdb_crop/imdbfilelist.txt', transformations, db='imdb')
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True, sampler=None)
test_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = 100, shuffle = False, num_workers = 8)
print("computing thresholds ", end='')
meta_data = open("/data_set_path/imdb_crop/imdbfilelist.txt")
all_vals = []
try:
    while True:
        line = meta_data.readline()
        vals = line.split("\t")
        try:
            all_vals.append(int(vals[1]))
        except:
            exit()
except:
    meta_data.close()
print("using " + str(len(all_vals)) + " data elements ", end='')
all_vals = torch.Tensor(all_vals)
sorted_values = torch.sort(all_vals)[0]
thresholds = [sorted_values[(i+1) * sorted_values.shape[0] // num_classes].item() for i in range(num_classes - 1)]
print("done")
print(thresholds)

def mutual_information(preds):
  diff = 0
  logits = preds
  logits_mean = logits.mean(0)
  entropy_y  = - (logits_mean * torch.log2(logits_mean)).sum()
  entropy_y_x = -(logits  * torch.log2(logits)).sum(1) 
  diff += (entropy_y_x - entropy_y).mean()
  return -diff



def calc_bins(preds,targets):
  # Assign each prediction to a bin
  labels_oneh = F.one_hot(targets,2)
  num_bins = 10
  bins = np.linspace(0.1, 1, num_bins)
  binned = np.digitize(preds, bins)

  # Save the accuracy, confidence and size of each bin
  bin_accs = np.zeros(num_bins)
  bin_confs = np.zeros(num_bins)
  bin_sizes = np.zeros(num_bins)

  for bin in range(num_bins):
    bin_sizes[bin] = len(preds[binned == bin])
    if bin_sizes[bin] > 0:
      bin_accs[bin] = (labels_oneh[binned==bin]).sum() / bin_sizes[bin]
      bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

  return bins, binned, bin_accs, bin_confs, bin_sizes

def get_metrics(preds,targets):
  ECE = 0
  MCE = 0
  bins, _, bin_accs, bin_confs, bin_sizes = calc_bins(preds,targets)

  for i in range(len(bins)):
    abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
    ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
    MCE = max(MCE, abs_conf_dif)

  return ECE, MCE
import matplotlib.patches as mpatches

def draw_reliability_graph(preds,targets):
  ECE, MCE = get_metrics(preds,targets)
  bins, _, bin_accs, _, _ = calc_bins(preds,targets)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.gca()

  # x/y limits
  ax.set_xlim(0, 1.05)
  ax.set_ylim(0, 1)

  # x/y labels
  plt.xlabel('Confidence')
  plt.ylabel('Accuracy')

  # Create grid
  ax.set_axisbelow(True) 
  ax.grid(color='gray', linestyle='dashed')

  # Error bars
  plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

  # Draw bars and identity line
  plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
  plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

  # Equally spaced axes
  plt.gca().set_aspect('equal', adjustable='box')

  # ECE and MCE legend
  ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
  MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
  plt.legend(handles=[ECE_patch, MCE_patch])

  plt.show()
  
  plt.savefig('calibrated_network.png', bbox_inches='tight')

def Stability(preds):
  all_std = 0 
  for i in range(num_classes):
      single_std = preds[:,i].std(dim = 0)
      all_std += single_std
  return 1-all_std/num_classes



def test(model, test_loader):
    model.eval()
    test_loss, accuracy, total_elts = 0, 0, 0
    all_ages = []
    all_features = []
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(test_loader):
            data, age = data.to(device), target.to(device)
            target = torch.zeros_like(age)
            for i in range(len(thresholds)):
                target[torch.where(age > thresholds[i])[0]] = i + 1

            features = model(data)
            output = model.linear(features).softmax(-1)

            all_ages.append(age)
            all_features.append(features.reshape(features.shape[0]))
            test_loss += criterion(output, target.long()).item()
            pred = output #.argmax(dim=1, keepdim=True)
            all_preds.append(pred.softmax(-1))
        
            total_elts += target.shape[0]
            all_targets.append(target)
            if total_elts >= 10000:
                break
        
    ages = torch.cat(all_ages).to("cpu") # all the ages associated with the samples
    features = torch.cat(all_features).to("cpu") # all the features
    preds = torch.cat(all_preds).to("cpu") # all the predictions
    targets = torch.cat(all_targets).to("cpu")
    print("Stability",Stability(preds))
    print("Mutual Information", mutual_information(preds))
    draw_reliability_graph(preds.detach().numpy(),targets.long())

test(model,test_loader)
