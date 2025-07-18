import json
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import numpy as np

input_file = gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz")
dataset = []
for l in input_file:
    d = eval(l)
    dataset.append(d)
input_file.close()

output_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/output"
os.makedirs(output_dir, exist_ok=True)  
df = pd.DataFrame(dataset)
df = df.drop(columns=["found_funny", "compensation"])

df = df.dropna(subset=["hours"])
df["hours"] = df["hours"].astype(float)
df["products"] = df["products"].fillna(0)

# Add engineered features
df["review_length"] = df["text"].astype(str).apply(lambda x: len(x.split()))
df["early_access"] = df["early_access"].astype(int)

# Meta feature list (cleaned)
meta_features = ["products", "review_length", "early_access"]


vectorizer = TfidfVectorizer(
    max_features=10000,      # Limit to top 10k words for performance
    stop_words="english",    # Remove common stopwords
    lowercase=True,
    min_df=5                 # Only include words in at least 5 reviews
)

X_text = vectorizer.fit_transform(df["text"].astype(str))
word_features = vectorizer.get_feature_names_out()

X_meta = df[meta_features].values
X_meta_scaled = StandardScaler().fit_transform(X_meta)  # scale for fairness

# Combine text + meta features
X_combined = hstack([csr_matrix(X_meta_scaled), X_text])
combined_feature_names = meta_features + list(word_features)

# ======================================
# SelectKBest with f_regression
# ======================================
y = df["hours"]
selector = SelectKBest(score_func=f_regression, k='all')
selector.fit(X_combined, y)

scores = selector.scores_
score_pairs = list(zip(combined_feature_names, scores))
sorted_scores = sorted(score_pairs, key=lambda x: -x[1])

# Output top and bottom 50 features
kbest_path = os.path.join(output_dir, "combined_feature_importance.txt")
with open(kbest_path, "w", encoding="utf-8") as f:
    f.write("Top 50 most important features (meta + text):\n\n")
    for name, score in sorted_scores[:50]:
        f.write(f"{name}: {score:.2f}\n")

    f.write("\nBottom 50 least informative features:\n\n")
    for name, score in sorted_scores[-50:]:
        f.write(f"{name}: {score:.2f}\n")

'''
# 3. Open report file in the correct directory
report_path = os.path.join(output_dir, "eda_report.txt")
with open(report_path, "w", encoding="utf-8") as report:

    # Dataset shape
    report.write(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n\n")

    # Column names
    report.write("Columns:\n")
    report.write(", ".join(df.columns) + "\n\n")

    # Data types
    report.write("Data types:\n")
    report.write(str(df.dtypes) + "\n\n")

    # Target Distribution (hours)
    report.write("Target variable: 'hours'\n")
    report.write(str(df["hours"].describe()) + "\n\n")

    # Check for skewed distribution
    skew = df["hours"].skew()
    report.write(f"Skewness of 'hours': {skew:.2f}\n\n")

    # Duplicate check
    num_duplicates = df.duplicated(subset=["username", "product_id", "text"]).sum()
    report.write(f"Number of duplicate reviews: {num_duplicates}\n\n")

    # Missing values
    missing_counts = df.isnull().sum()
    report.write("Missing values per column:\n")
    report.write(str(missing_counts) + "\n\n")

    # Percent missing
    report.write("Percentage of missing values:\n")
    report.write(str((df.isnull().mean() * 100).round(2)) + "\n\n")

    # Correlation with target
    numeric_corr = df.corr(numeric_only=True)
    if "hours" in numeric_corr.columns:
        corr_with_hours = numeric_corr["hours"].sort_values(ascending=False)
        report.write("Correlation with target 'hours':\n")
        report.write(str(corr_with_hours) + "\n\n")
    else:
        report.write("Target variable 'hours' not found in numeric correlations.\n\n")

    # Check for target leakage in text
    sample_texts = df["text"].dropna().sample(1000, random_state=42)
    time_mention_count = sum(bool(re.search(r"\d+\s*(h|hour)", txt.lower())) for txt in sample_texts)
    report.write(f"Potential target leakage (review mentions hours): {time_mention_count}/1000 sampled texts\n\n")

# 4. Save the histogram to the output directory
plot_path = os.path.join(output_dir, "hours_distribution.png")
plt.figure(figsize=(10, 5))
plt.hist(df["hours"], bins=100, edgecolor='black')
plt.title("Distribution of 'hours' Played")
plt.xlabel("Hours")
plt.ylabel("Frequency")
plt.xlim(0, 100)  # Trim for readability
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
'''