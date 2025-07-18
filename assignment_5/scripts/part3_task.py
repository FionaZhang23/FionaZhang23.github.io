import gzip
import numpy as np
import csv
import utils
import pandas as pd
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, mean, lit, isnan, when, count
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType
import math

with gzip.open("/deac/csc/classes/csc373/data/assignment_5/steam_reviews.json.gz", 'rt', encoding='utf-8') as f:
    dataset = [eval(line) for line in f]

# Create DataFrame with required columns
processed_data = []
for record in dataset:
    hours = record.get('hours')
    if hours is not None:
        processed_data.append({
            'username': record.get('username'),
            'product_id': record.get('product_id'),
            'hours': float(hours)
        })

pdf = pd.DataFrame(processed_data)
pdf.dropna(subset=['username', 'product_id', 'hours'], inplace=True)

# Discretize hours into 1-5 bins
pdf = utils.discretize_hours(pdf, n_bins=5, strategy='quantile')
sampled_df = utils.sample_for_testing(pdf)

# Save to CSV and load as Spark DataFrame
output_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/data"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "steam_reviews_recommendation.csv")
sampled_df.to_csv(csv_path, index=False)


spark = SparkSession.builder.appName("SteamRecommendationComparison").getOrCreate()

# 1. Load the preprocessed CSV file
data_path = "/deac/csc/classes/csc373/zhanx223/assignment_5/data/steam_reviews_recommendation.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True)
df = (df
      .withColumn("hours", df["hours"].cast(DoubleType()))
      .withColumn("hours_bin", df["hours_bin"].cast(DoubleType()))
      .na.drop(subset=["hours", "hours_bin"])  # Drop rows with null values
     )

# Verify data quality
print("Data quality check:")
df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in ["hours", "hours_bin"]]).show()

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# 2. Evaluation function
def evaluate_predictions(predictions, label_col, model_name, results):
    evaluator = RegressionEvaluator(
        labelCol=label_col, 
        predictionCol="prediction", 
        metricName="mse"
    )
    mse = evaluator.evaluate(predictions)
    
    over = predictions.filter(col("prediction") > col(label_col)).count()
    under = predictions.filter(col("prediction") < col(label_col)).count()
    total = predictions.count()
    
    results.append({
        'model': model_name,
        'mse': float(mse),
        'overpredictions': over,
        'underpredictions': under,
        'total_predictions': total,
        'target_type': label_col
    })

# 3. Prepare StringIndexers (common for both ALS models)
user_indexer = StringIndexer(
    inputCol="username", 
    outputCol="userIndex", 
    handleInvalid="skip"
)
item_indexer = StringIndexer(
    inputCol="product_id", 
    outputCol="itemIndex", 
    handleInvalid="skip"
)

# 4. Run all three prediction tasks
results = []

# Task 1: Dummy regressor (mean prediction) with raw hours
mean_hours = train_df.select(mean("hours")).collect()[0][0]
dummy_pred = test_df.withColumn("prediction", lit(mean_hours))
evaluate_predictions(dummy_pred, "hours", "DummyRegressor-RawHours", results)

# Task 2: ALS with raw hours
als_hours = ALS(
    userCol="userIndex",
    itemCol="itemIndex",
    ratingCol="hours",
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)

pipeline_hours = Pipeline(stages=[user_indexer, item_indexer, als_hours])
model_hours = pipeline_hours.fit(train_df)
pred_hours = model_hours.transform(test_df).dropna(subset=["prediction"])
evaluate_predictions(pred_hours, "hours", "ALS-RawHours", results)

# Task 3: ALS with discretized hours (1-5)
als_bins = ALS(
    userCol="userIndex",
    itemCol="itemIndex",
    ratingCol="hours_bin",
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)

pipeline_bins = Pipeline(stages=[user_indexer, item_indexer, als_bins])
model_bins = pipeline_bins.fit(train_df)
pred_bins = model_bins.transform(test_df).dropna(subset=["prediction"])
evaluate_predictions(pred_bins, "hours_bin", "ALS-Discretized", results)

output_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/output"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
results_path = os.path.join(output_dir, "recommendation_report.txt")

with open(results_path, "w") as f:
    f.write("=== Recommendation Model Evaluation Results ===\n\n")
    for result in results:
        f.write(f"Model: {result['model']}\n")
        f.write(f"Target: {result['target_type']}\n")
        f.write(f"MSE: {result['mse']:.4f}\n")
        f.write(f"Overpredictions: {result['overpredictions']}\n")
        f.write(f"Underpredictions: {result['underpredictions']}\n")
        f.write(f"Total predictions: {result['total_predictions']}\n")
        f.write("\n" + "="*50 + "\n\n")

print(f"Results saved to: {results_path}")
spark.stop()