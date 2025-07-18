"""
Acknowledgments:
- This script was developed using references and inspiration from:
  1. code_snippets.py
"""
import os
import numpy as np
import torch
import PIL.Image
from tqdm import tqdm
from sklearn import metrics
from transformers import AutoFeatureExtractor, ResNetForImageClassification

def get_file_paths(directory):
    files = []
    for i in os.listdir(directory):
        if i.endswith('.JPEG'):
            files.append(os.path.join(directory, i))
    return sorted(files)

def get_true_labels(directory, test_files):
    anomolous_files = []
    with open('../data/ground_truth.txt') as f:
        for line in f.readlines():
            anomolous_files.append(os.path.join(directory, line.strip()))
    return np.array([i in anomolous_files for i in test_files])

def assign_random_scores(n):
    return np.random.uniform(size=n)

def evaluate(true_labels, predicted_scores):
    return metrics.roc_auc_score(true_labels, predicted_scores)

def get_average_color(train_files):
    train_mean_pixel = np.zeros(3)
    for path in train_files:
        train_mean_pixel += np.asarray(PIL.Image.open(path).convert('RGB')).mean(axis=(0, 1))
    train_mean_pixel /= len(train_files)
    return train_mean_pixel

def compute_histograms(file_paths):
    all_hists = []
    for path in file_paths:
        x = np.asarray(PIL.Image.open(path).convert('RGB').resize((256, 256)))
        hists = [np.histogram(x[:, :, c], bins=32, range=(0, 256))[0] for c in range(3)]
        all_hists.append(np.concatenate(hists))
    return np.stack(all_hists)


