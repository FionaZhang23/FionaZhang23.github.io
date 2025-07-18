"""
===============================================================================
Developed by: Fiona Zhang
Institution: Wake Forest University
Course: CSC373 Data Mining

Acknowledgements:
- This script was developed using references, inspiration and support from:
  1. DeepSeek
  2. scikit-learn (sklearn) library examples
  3. DEAC HPC
  4. PySpark API
===============================================================================
"""
import os
import pandas as pd
import utils
from sklearn.svm import SVR
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
import joblib
import numpy as np
data_dir = "/deac/csc/classes/csc373/data/assignment_5"
output_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/output"
cleandata_dir = "/deac/csc/classes/csc373/zhanx223/assignment_5/data"
data_path = os.path.join(data_dir, "steam_reviews.json.gz")
cleandata_path = os.path.join(cleandata_dir, "steam_reviews_cleaned.csv")
recommendation_data_path = "/deac/csc/classes/csc373/zhanx223/assignment_5/data/steam_reviews_recommendation.csv"
regression_pipeline_path = os.path.join(output_dir, "regression_pipeline.joblib")
classification_pipeline_path = os.path.join(output_dir, "classification_pipeline.joblib")
recommendation_pipeline_path = "/deac/csc/classes/csc373/zhanx223/assignment_5/output/recommendation_pipeline"

'''
texts = utils.parse_json_file(data_path)
hours = utils.get_hours_from_json(data_path)
df = pd.DataFrame({'text': texts, 'hours': hours})
'''
df = utils.load_csv(cleandata_path)

#Regression Task
train_df, dev_df = utils.split_data(df)
train_df_cleaned = utils.remove_outliers(df, column="hours", quantile=0.9)

X_train = train_df_cleaned["text"]
y_train = utils.log_transform_target(train_df_cleaned["hours"])
X_dev = dev_df['text']
y_dev = utils.log_transform_target(dev_df["hours"])

regression_pipeline = Pipeline([
    ("transformer", TfidfVectorizer(
        analyzer='word',
        max_features=10000,
        ngram_range=(1, 3),
        min_df=5,
        max_df=0.95,
        stop_words='english'
    )),
    ("regressor", SVR())
])
regression_pipeline.fit(X_train, y_train)
joblib.dump(regression_pipeline, regression_pipeline_path)
print(f"Pipeline saved to: {regression_pipeline_path}")

reg_pipeline = joblib.load(regression_pipeline_path)
predictions_log = reg_pipeline.predict(X_dev)
results = utils.reg_evaluate_preds(y_dev, predictions_log)
print(results)
'''
#transform back to origianl scale for playing hours(if needed)
predictions = utils.inverse_log_transform(predictions_log)
'''
#Classification Task (median = 15.3)
median_hours = df["hours"].median()
print(median_hours)
df["labels"] = (df["hours"] > df["hours"].median()).astype(int)
df = df[["text", "hours", "year", "labels"]]
train_df, dev_df = utils.split_data(df)
X_train = df["text"]
y_train = df["labels"]
X_dev = dev_df['text']
y_dev = dev_df["labels"]

classification_pipeline = Pipeline([
        ("transformer", TfidfVectorizer(
            analyzer='word',
            max_features=10000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.95,
            stop_words='english'
        )),
        ("clf", LogisticRegression(max_iter=1000))
    ])
classification_pipeline.fit(X_train, y_train)
joblib.dump(classification_pipeline, classification_pipeline_path)
print(f"Pipeline saved to: {classification_pipeline_path}")

clf_pipeline = joblib.load(classification_pipeline_path)
clf_prediction = clf_pipeline.predict(X_dev)
results = utils.clf_evaluate_preds(y_dev, clf_prediction)
print(results)

#Recommendation Task
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log, lit, rand, pow
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.appName("LoadPredictionData").getOrCreate()

df = spark.read.csv(recommendation_data_path, header=True, inferSchema=True)
train_df, dev_df = df.randomSplit([0.8, 0.2], seed=42)

user_indexer = StringIndexer(inputCol="username", outputCol="userIndex", handleInvalid="skip")
product_indexer = StringIndexer(inputCol="product_id", outputCol="itemIndex", handleInvalid="skip")

als = ALS(
    userCol="userIndex",
    itemCol="itemIndex",
    ratingCol="hours_bin",
    coldStartStrategy="drop",
    nonnegative=True,
    seed=42
)

# Build a pipeline that includes indexing and the ALS model.
recommendation_pipeline = Pipeline(stages=[user_indexer, product_indexer, als])
recommendation_model = recommendation_pipeline.fit(df)
os.makedirs(os.path.dirname(recommendation_pipeline_path), exist_ok=True)
recommendation_model.write().overwrite().save(recommendation_pipeline_path)
print(f"Pipeline model saved to: {recommendation_pipeline_path}")

loaded_model = PipelineModel.load(recommendation_pipeline_path)
predictions = loaded_model.transform(dev_df)

evaluator = RegressionEvaluator(labelCol="hours_bin", predictionCol="prediction", metricName="mse")
mse = evaluator.evaluate(predictions)
print(f"MSE: {mse}")
overpredicted = predictions.filter(predictions.prediction > predictions.hours_bin).count()
underpredicted = predictions.filter(predictions.prediction < predictions.hours_bin).count()
print(f"Overpredicted instances: {overpredicted}")
print(f"Underpredicted instances: {underpredicted}")

#transform prediction to orginal scale if needed
predictions = predictions.withColumn("predicted_hours", pow(lit(3.0), col("prediction")) - 1)

spark.stop()
