"""
Acknowledgments:
- This script was developed using references, inspiration and support from:
  1. DeepSeek
  2. scikit-learn (sklearn) library examples
  3. DEAC HPC
"""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
from utils import load_data, decode_labels, reshape_scans, reshape_labels
import os
from sklearn.preprocessing import OneHotEncoder
from custom_classifier import OneHotClassifier

data_path = "/deac/csc/classes/csc373/data/assignment_4"      
output_path = "/deac/csc/classes/csc373/zhanx223/assignment_4/output"
model_path = "/deac/csc/classes/csc373/zhanx223/assignment_4/output/modeling_pipeline.pkl"

scan_fall_2019 = os.path.join(output_path, "filtered_scans.npy")
labels_fall_2019 = os.path.join(output_path, "filtered_labels.npy")
scan_spring_2025 = os.path.join(data_path, "scan_spring_2025.npy")

predictions_path = os.path.join(output_path, "predictions.npy")
#load datasets

scans = load_data(scan_fall_2019)
labels = load_data(labels_fall_2019)
spring_scans = load_data(scan_spring_2025)

#reshape the data for test purpose
scans_reshaped = reshape_scans(scans)
spring_scans_reshaped = reshape_scans(spring_scans)

X_train, X_test, y_train, y_test = train_test_split(scans, labels, test_size=0.2, random_state=48)

#Dummy Classifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred = dummy_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Dummy Classifier Accuracy: {accuracy:.2f}")

#Baseline Classsifier
lg_pipeline =  Pipeline([("classifier", LogisticRegression())])
lg_pipeline.fit(X_train, y_train)
y_pred_lg = lg_pipeline.predict(X_test)
lg_accuracy = accuracy_score(y_test, y_pred_lg)
print(f"Baseline LogisticRegression Accuracy: {lg_accuracy:.2f}")

#Advanced model for final prediction
rm_pipeline = Pipeline([
  ("classifier", OneHotClassifier())
])
rm_pipeline.fit(scans, labels)
joblib.dump(rm_pipeline, model_path)

pipeline = joblib.load(model_path)
prediction = pipeline.predict(spring_scans)
np.save(predictions_path, prediction)