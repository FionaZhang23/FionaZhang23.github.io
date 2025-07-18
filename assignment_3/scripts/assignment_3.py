"""
Acknowledgments:
- This script was developed using references and inspiration from:
  1. DeepSeek
  2. scikit-learn (sklearn) library examples
  3. code_snippets.py
"""
import numpy as np
import torch
import utils
from transformers import AutoFeatureExtractor, ResNetForImageClassification
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import joblib
from PIL import Image
import os
from sklearn.pipeline import Pipeline
from Transformer import CustomTransformer

# Define directories
train_data_dir = '/deac/csc/classes/csc373/data/assignment_3/train'
dev_data_dir = '/deac/csc/classes/csc373/data/assignment_3/dev'
output_dir = '/deac/csc/classes/csc373/zhanx223/assignment_3/output'
report_path = os.path.join(output_dir, 'output_report.txt')
pipeline_path = os.path.join(output_dir, 'final_pipeline.pkl')

train_files = utils.get_file_paths(train_data_dir)
dev_files = utils.get_file_paths(dev_data_dir)
ground_truth = utils.get_true_labels(dev_data_dir, dev_files)

X_train = [Image.open(i).convert("RGB") for i in train_files]
X_test = [Image.open(i).convert("RGB") for i in dev_files]

# 1. Dummy Detector (Random Scores)
random_ood_scores = utils.assign_random_scores(len(dev_files))
auc_dummy = utils.evaluate(ground_truth, random_ood_scores)
print(f'Dummy Detector AUC: {round(auc_dummy, 2)}')

# 2. Baseline Detector (Average RGB Distance)
average_train_color = utils.get_average_color(train_files)
predicted_scores_baseline = [np.linalg.norm(utils.get_average_color([img]) - average_train_color) for img in dev_files]
auc_baseline = utils.evaluate(ground_truth, predicted_scores_baseline)
print(f'Baseline Detector AUC: {round(auc_baseline, 2)}')

# 3. Best Model
#Testing the IsolationForest Model Prediction Score
'''
pipeline = Pipeline([
    ('featurizer', CustomTransformer()), 
    ('detector', IsolationForest(contamination=0.1, random_state=42)),
])
pipeline.fit(X_train)
predictions = -pipeline.decision_function(X_test)
auc_iso = utils.evaluate(ground_truth,predictions)
print(f'IsolationForest Detector AUC: {round(auc_iso, 2)}')
'''
best_performing_pipeline = Pipeline([
    ('featurizer', CustomTransformer()), 
    ('detector', OneClassSVM(nu=0.1))
])
best_performing_pipeline.fit(X_train)
joblib.dump(best_performing_pipeline, pipeline_path)
print(f'Pipeline saved to {pipeline_path}')

best_performing_pipeline = joblib.load(pipeline_path)
predictions = -best_performing_pipeline.decision_function(X_test)
auc_ocsvm = utils.evaluate(ground_truth, predictions)
print(f'One Class SVM Detector AUC: {round(auc_ocsvm, 2)}')
'''
summary = f"""
Output Report
-------------
Dummy Detector AUC: {round(auc_dummy, 2)}
Baseline Detector AUC: {round(auc_baseline, 2)}')
IsolationForest Detector AUC: {round(auc_iso, 2)}')
One-Class SVM Detector AUC(Embeddings): {round(auc_ocsvm, 2)}
Pipeline saved at: {pipeline_path}
"""

with open(report_path, 'w') as f:
    f.write(summary)

print(f'Report saved to {report_path}')
'''