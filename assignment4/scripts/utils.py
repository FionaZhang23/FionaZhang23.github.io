import numpy as np

def load_data(data_path):

    data = np.load(data_path)
    return data

def decode_labels(one_hot_labels):

    return np.argmax(one_hot_labels, axis=-1)

def reshape_scans(scans):
    scans_reshaped = scans.reshape(-1, 1) 
    return scans_reshaped

def reshape_labels(labels):
    labels_reshaped = labels.reshape(-1)
    return labels_reshaped