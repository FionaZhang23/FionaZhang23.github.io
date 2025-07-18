import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import utils
from transformers import CustomTransformer

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



np.random.seed(0)

def run_housing_pipeline():
    meta_data_path = '../data/feature_descriptions.csv'
    train_data_path = "/deac/csc/classes/csc373/data/housing/train.csv"
    test_data_path = "/deac/csc/classes/csc373/data/housing/dev.csv"

    train_data, train_labels = utils.read_csv_data(train_data_path, "SalePrice")
    test_data, test_labels = utils.read_csv_data(test_data_path, "SalePrice")
    numeric_cols, categoric_cols = utils.get_meta_data(meta_data_path)



    names = ['Baseline', 'SGDR']
    regressors = [DummyRegressor(strategy='mean'), SGDRegressor()]

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categoric_cols)
        ]
    )
    
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("leakage_remover", CustomTransformer()),
            ("regressor", None)
        ]
    )
    
    for i in range(len(names)):
        pipeline.set_params(regressor=regressors[i])
        pipeline.fit(train_data, train_labels)
        predictions = pipeline.predict(test_data)
        print(f"{names[i]}: R^2 score: %.3f" % r2_score(test_labels, predictions))

#for part 4
def run_book_recommender_pipeline_shorter():
    input_path = "/deac/csc/classes/csc373/data/fantasy/fantasy_100.json"
    output_path = "../output/fantasy_modeling_report(part4).txt"
    
    train_data, train_labels = utils.read_json_data(input_path)
    train_labels = np.array(train_labels)
    train_data = list(map(str, train_data))

    names = ['Baseline', 'SGDR','DecisionTree']
    regressors = [
        DummyRegressor(strategy='mean'),
        SGDRegressor(),
        DecisionTreeRegressor()
        ]

    pipeline = Pipeline([ 
        ("vectorizer", TfidfVectorizer()),   
        ("regressor", None)           
    ])

    file_handle = open(output_path, mode="w", encoding="utf-8")
    for i in range(len(names)):
        pipeline.set_params(regressor=regressors[i])
        pipeline.fit(train_data, train_labels)
        predictions = pipeline.predict(train_data)
        file_handle.write(f"{names[i]}: R^2 score: %.3f\n" % r2_score(train_labels, predictions))
    file_handle.close()  

#for part 5
def run_book_recommender_pipeline():
    input_path = "/deac/csc/classes/csc373/data/fantasy/fantasy_10000.json"
    output_path = "../output/fantasy_modeling_report.txt"
    
    train_data, train_labels = utils.read_json_data(input_path)
    train_labels = np.array(train_labels)
    train_data = list(map(str, train_data))

    X_train, X_test, y_train, y_test = train_test_split(list(train_data), train_labels, test_size=0.3, random_state=42)

    names = ['Baseline','LinearRegression', 'RandomForest', 'SVR', 'LinearSVR']
    regressors = [
        DummyRegressor(strategy='mean'),
        LinearRegression(),
        RandomForestRegressor(),
        SVR(kernel='rbf', C=1.5),
        LinearSVR()
    ]
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(
            analyzer='word',
            stop_words='english',
            ngram_range=(1,3)
        )),
        ("regressor", None)  
    ])

    file_handle = open(output_path, mode="w", encoding="utf-8")
    for i in range(len(names)):
        pipeline.set_params(regressor=regressors[i])
        pipeline.fit(X_train, y_train)
        prediction_1 = pipeline.predict(X_train)
        predictions = pipeline.predict(X_test)
        file_handle.write(f"{names[i]}: R^2 score for test dataset: %.3f\n" % r2_score(y_test, predictions))
        file_handle.write(f"{names[i]}: R^2 score for train dataset: %.3f\n" % r2_score(y_train, prediction_1))
    file_handle.close()  

    
#run_housing_pipeline()
#run_book_recommender_pipeline_shorter()
run_book_recommender_pipeline()
