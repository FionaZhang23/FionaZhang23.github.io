import json
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def get_meta_data(file_path):
    descriptions = pd.read_csv(file_path)
    numeric_cols = descriptions[descriptions['type'] == 'numeric'].name.to_list()
    categoric_cols = descriptions[descriptions['type'] == 'category'].name.to_list()

    return numeric_cols, categoric_cols

def read_csv_data(file_path, label_col):
    data = pd.read_csv(file_path, header=0)
    labels = data[label_col]
    features = data.drop(label_col, axis=1)
    return features, labels

def clean_text(text):
    
    text = re.sub(r"\n", " ", text)  # Replace newlines with spaces
    return text

def read_json_data(file_path):
    file_handle = open(file_path)
    data = []
    for line in file_handle:
        d = json.loads(line)
        data.append(d)
    file_handle.close()

    labels = np.asarray([d['rating'] for d in data])
    features = [clean_text(d['review_text']) for d in data]
    features = np.asarray(features).reshape(-1,1)

    return features, labels