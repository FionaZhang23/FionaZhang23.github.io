import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import utils

data_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/data"
output_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/output"
file_path = os.path.join(data_dir, "steam_reviews_cleaned.csv")
report_path = os.path.join(output_dir, "classification_report.txt") 


df = utils.load_csv(file_path)
print(f"✅ Loaded dataset with {len(df)} rows.")

if "year" not in df.columns:
    df["year"] = df["date"].apply(lambda x: int(str(x)[:4])) if "date" in df.columns else None

if df["year"].isnull().any():
    df = df.dropna(subset=["year"])

median_hours = df["hours"].median()
df["label"] = (df["hours"] > median_hours).astype(int)
print(f"📊 Median hours: {median_hours:.2f}")

df = df[["text", "hours", "year", "label"]]

train_pre2015 = df[df["year"] <= 2014].reset_index(drop=True)
test_post2014 = df[df["year"] >= 2015].reset_index(drop=True)

train_post2014 = test_post2014.copy()
test_pre2015 = train_pre2015.copy()

models = [
    ("DummyClassifier", DummyClassifier(strategy='most_frequent')),
    ("DecisionTree (Baseline)", DecisionTreeClassifier(random_state=42)),
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("SGDClassifier", SGDClassifier(max_iter=1000, tol=1e-3, random_state=42))]

X_train_pre2015 = train_pre2015["text"]
y_train_pre2015 = train_pre2015["label"]
X_test_post2014 = test_post2014["text"]
y_test_post2014 = test_post2014["label"]

X_train_post2014 = train_post2014["text"]
y_train_post2014 = train_post2014["label"]
X_test_pre2015 = test_pre2015["text"]
y_test_pre2015 = test_pre2015["label"]

# Define a simple evaluation function
def evaluate_preds(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    overpredicts = ((y_pred == 1) & (y_true == 0)).sum()
    underpredicts = ((y_pred == 0) & (y_true == 1)).sum()
    return accuracy, overpredicts, underpredicts

# List to store the report lines
report_lines = []
report_lines.append("Classification Report\n" + "="*80 + "\n")

# Loop over each model and run both experiments
for name, model in models:
    report_lines.append(f"\n🚀 Evaluating Model: {name}\n" + "-"*80)
    
    # Create a pipeline using TF-IDF vectorization and the current classifier
    pipeline = Pipeline([
        ("transformer", TfidfVectorizer(
            analyzer='word',
            max_features=10000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.95,
            stop_words='english'
        )),
        ("clf", model)
    ])
    
    # Experiment 1: Train on data with year ≤ 2014, test on data with year ≥ 2015
    pipeline.fit(X_train_pre2015, y_train_pre2015)
    preds1 = pipeline.predict(X_test_post2014)
    acc1, over1, under1 = evaluate_preds(y_test_post2014, preds1)
    exp1_report = (f"Setup 1 (Train ≤2014 → Test ≥2015):\n"
                   f"  Accuracy: {acc1:.4f}\n"
                   f"  Overpredicts: {over1}\n"
                   f"  Underpredicts: {under1}\n")
    report_lines.append(exp1_report)
    
    # Experiment 2: Train on data with year ≥ 2015, test on data with year ≤ 2014
    pipeline.fit(X_train_post2014, y_train_post2014)
    preds2 = pipeline.predict(X_test_pre2015)
    acc2, over2, under2 = evaluate_preds(y_test_pre2015, preds2)
    exp2_report = (f"Setup 2 (Train ≥2015 → Test ≤2014):\n"
                   f"  Accuracy: {acc2:.4f}\n"
                   f"  Overpredicts: {over2}\n"
                   f"  Underpredicts: {under2}\n")
    report_lines.append(exp2_report)


report = "\n".join(report_lines)

print(report)

with open(report_path, "w") as f:
    f.write(report)
print(f"\n📄 Classification report saved to: {report_path}")
