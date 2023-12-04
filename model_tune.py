import os
import time
import pandas as pd
from pyspark.ml import Pipeline
from argparse import ArgumentParser
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
# from pyspark.sql import functions as F
from pyspark.ml.tuning import (
    ParamGridBuilder,
    CrossValidator
)

def parse_args():
    parser = ArgumentParser(description='Recommendation System Model Tuning')
    parser.add_argument('--memory', type=str, default='4')
    parser.add_argument('--num_threads', type=str, default='1')
    parser.add_argument('--data_path', type=str, required=True)
    return parser.parse_known_args()[0]


def train(args):
    memory_limit = args.memory
    num_threads = args.num_threads
    data_path =  args.data_path
    spark = SparkSession.builder \
    .master(f'local[{num_threads}]') \
    .config("spark.driver.memory", f"{memory_limit}g") \
    .appName('RecommendationSystem') \
    .getOrCreate()

    # Load the dataset from CSV
    data = spark.read.csv(data_path, header=True, inferSchema=True)
    data = data.select(data['productID'],data['rating'],data['reviewerID'])

    # String Indexer for productId and ReviewerId
    indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(data.columns)-set(['rating'])) ]
    pipeline = Pipeline(stages=indexer)
    transformed = pipeline.fit(data).transform(data)
    # print("Transformed Data")

    # Split the Dataset into train test slpit
    (training,test)=transformed.randomSplit([0.8, 0.2], seed=0)

    # print("Training Size", training.count())
    # print("Test Size", test.count())
    # print("splitting Done!")

    # ALS model Training
    start_time = time.time()
    als=ALS(maxIter=10,
            regParam=0.09,
            rank=100,
            userCol="reviewerID_index",
            itemCol="productID_index",
            ratingCol="rating",
            oldStartStrategy="drop",
            nonnegative=True)
    param_grid = ParamGridBuilder()\
             .addGrid(als.rank, [1, 20, 40])\
             .addGrid(als.maxIter, [1, 5, 10])\
             .addGrid(als.regParam, [0.05, 0.5, 0.8, 0.9])\
             .build()
    
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

    cv = CrossValidator(
            estimator=als,
            estimatorParamMaps=param_grid,
            evaluator=evaluator,
            numFolds=3)

    model = cv.fit(training)

    best_model = model.bestModel
    print('rank: ', best_model.rank)
    print('MaxIter: ', best_model._java_obj.parent().getMaxIter())
    print('RegParam: ', best_model._java_obj.parent().getRegParam())
    print("Total Time : ", time.time()-start_time)
    

if __name__ == '__main__':
    args = parse_args()
    print(args)
    train(args)