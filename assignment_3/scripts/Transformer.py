"""
Acknowledgments:
- This script was developed using references and inspiration from:
  1. code_snippets.py
"""
import numpy as np
import torch
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.featurizer = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        if torch.cuda.is_available():
            model = model.to('cuda:0')
        self.batch_size = 50

    def fit(self, X, y=None):
        return self

    def chunks(self, l, n):
        return [l[i:i+n] for i in range(0, len(l), n)]

    def transform(self, X):
        embeddings = None
        for chunk in self.chunks(X, self.batch_size): 
            inputs = self.featurizer(chunk, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.to('cuda:0')
            with torch.no_grad():
                outputs = self.model(**inputs)
            embedding = outputs.logits.cpu().numpy()
            if embeddings is None:
                embeddings = embedding
            else:
                embeddings = np.vstack([embeddings, embedding])
        return embeddings