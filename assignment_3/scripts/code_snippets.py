import os, tqdm
import torch
import numpy as np
import PIL.Image
from transformers import AutoFeatureExtractor, ResNetForImageClassification

from sklearn import metrics

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
            line = line.strip()
            anomolous_files.append(os.path.join(directory, line))
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
        hists = []
        for c in range(3):
            hist, _ = np.histogram(x[:,:,c], bins=32, range=(0, 256))
            hists.append(hist)
        hist = np.concatenate(hists)
        all_hists.append(hist)
    return np.stack(all_hists)

def chunks(l, n):
    return [l[i:i+n] for i in range(0, len(l), n)]

def compute_embeddings(file_paths, model, featurizer, batch_size=50):
    embeddings = None
    for chunk in chunks(file_paths, batch_size):
        images = [PIL.Image.open(i).convert('RGB') for i in chunk]
        inputs = featurizer(images, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda:0')
        with torch.no_grad():
            outputs = model(**inputs)
        embedding = outputs.logits.cpu().numpy()
        if embeddings is None:
            embeddings = embedding
        else:
            embeddings = np.vstack([embeddings, embedding])
    return embeddings

#--------- sample function calls ---------------------
train_data_dir = '/deac/csc/classes/csc373/data/assignment_3/train'
dev_data_dir = '/deac/csc/classes/csc373/data/assignment_3/dev'

train_files = get_file_paths(train_data_dir)
dev_files = get_file_paths(dev_data_dir)

ground_truth = get_true_labels(dev_data_dir, dev_files)

# 1. fast
random_ood_scores = np.random.uniform(size=len(dev_files))
auc = evaluate(ground_truth, random_ood_scores)
print(f'DummyDetector: {round(auc,2)}')

# 2. slow
average_train_color = get_average_color(train_files)
print(f'Average RGB color of training images is: {average_train_color}')

# 3. slower
# train_histogram_features = compute_histograms(train_files)
# print(f'Histogam of the first training image is:\n{train_histogram_features[0]}')

# 4. very slow - must use HPC, approximate wait time is 2 hours
# output will be written to slurm/OUTPUT-DDDDDDD.o
# feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
# model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
# if torch.cuda.is_available():
#    model = model.to('cuda:0')

# dev_embeddings = compute_embeddings(dev_files, model, feature_extractor)
# print(f'Embedding dimension is {dev_embeddings.shape[1]}')
# print(f'Embedding of the first dev image is:\n{dev_embeddings[0]}')
