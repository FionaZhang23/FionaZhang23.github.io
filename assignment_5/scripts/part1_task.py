import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import utils

# Define paths
data_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/data"
output_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/output"
data_path = os.path.join(data_dir, "steam_reviews_cleaned.csv")
test_sample_path = os.path.join(data_dir, "reviews_sample_1000.csv")
results_path = os.path.join(output_dir, "regression_report.txt")

print("🔄 Loading full cleaned dataset...")
df = utils.load_csv(data_path)
print(f"✅ Loaded {len(df)} rows.")

print("✂️ Splitting data 80/20...")
train_df, dev_df = utils.split_data(df)
print(f"✅ Train size: {len(train_df)}")
print(f"✅ Dev size: {len(dev_df)}")

print("📉 Removing top 10% outliers from training set...")
train_df_cleaned = utils.remove_outliers(train_df, column="hours", quantile=0.9)
print(f"✅ Training set after outlier removal: {len(train_df_cleaned)} rows.")

# Use log-transformed hours as target
X_train = train_df_cleaned["text"]
y_train = np.log2(train_df_cleaned["hours"] + 1)
X_dev = dev_df['text']
y_dev = np.log2(dev_df["hours"] + 1)

# Define the regression models
models = [
    ("DummyRegressor", DummyRegressor(strategy='mean')),
    ("LinearRegression(Baselie)", LinearRegression()),
    ("GradientBoosting", GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)),
    ("SGDRegressor", SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)),
    ("MultinomialNB", MultinomialNB()),
    ("SVR", SVR())
]

results_log = []

for name, model in models:
    print(f"\nTraining model: {name}")
    pipeline = Pipeline([
        ("transformer", TfidfVectorizer(
            analyzer='word',
            max_features=10000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.95,
            stop_words='english'
        )),
        ("regressor", model)
    ])

    # For MultinomialNB, discretize the target since it's a classifier
    if name == "MultinomialNB":
        y_train_discrete = np.round(y_train).astype(int)
        pipeline.fit(X_train, y_train_discrete)
        y_pred = pipeline.predict(X_dev)
    else:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_dev)

    # Evaluate predictions using your provided evaluate_predictions function
    results = utils.reg_evaluate_preds(y_dev, y_pred)
    print(results)
    results_str = f"Model: {name}\n" + "\n".join(f"{k}: {v:.4f}" for k, v in results.items())
    results_log.append(results_str)

# Save results to file
with open(results_path, "w") as f:
    f.write("\n".join(results_log))

print(f"\nComparison results saved to: {results_path}")
