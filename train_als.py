import os
import sys

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml import Pipeline
from pyspark.sql import functions as F
from datetime import datetime


MEMORY = "100"
NUM_CORES = "12"
MODEL_PATH = "output/als"
CSV_PATH = 'predictions.csv'

spark = SparkSession.builder \
    .master('local[*]') \
    .config("spark.driver.memory", f"{MEMORY}g") \
    .config("spark.cores.max", f"{NUM_CORES}") \
    .appName('my-cool-app') \
    .getOrCreate()

# Load the dataset from CSV
data = spark.read.csv("data/data.csv", header=True, inferSchema=True)

data = data.select(data['productID'],data['rating'],data['reviewerID'])
data.show()

# String Indexer for productId and ReviewerId
indexer = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in list(set(data.columns)-set(['rating'])) ]
pipeline = Pipeline(stages=indexer)
transformed = pipeline.fit(data).transform(data)
print("Transformed Data")
transformed.show(20)

# Split the Dataset into train test slpit
(training,test)=transformed.randomSplit([0.8, 0.2], seed=0)
training = training.limit(200)
print("Training Size", training.count())
print("splitting Done!")

# ALS model Training
als=ALS(maxIter=1,regParam=0.09,rank=25,userCol="reviewerID_index",itemCol="productID_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
model=als.fit(training)

# Save the model
# model.save("output/als")

# Evaluate the model for RMSE
evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
predictions=model.transform(test)
rmse=evaluator.evaluate(predictions)
print("RMSE="+str(rmse))
predictions.show()



# Save the predictions
recs=model.recommendForAllUsers(10).toPandas()
nrecs=recs.recommendations.apply(pd.Series) \
            .merge(recs, right_index = True, left_index = True) \
            .drop(["recommendations"], axis = 1) \
            .melt(id_vars = ['reviewerID_index'], value_name = "recommendation") \
            .drop("variable", axis = 1) \
            .dropna() 
nrecs=nrecs.sort_values('reviewerID_index')
nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['reviewerID_index']], axis = 1)
nrecs.columns = [
        
        'ProductID_index',
        'Rating',
        'UserID_index'
       
     ]
md=transformed.select(transformed['reviewerID'],transformed['reviewerID_index'],transformed['ProductID'],transformed['ProductID_index'])
md=md.toPandas()
dict1 =dict(zip(md['reviewerID_index'],md['reviewerID']))
dict2=dict(zip(md['ProductID_index'],md['ProductID']))
nrecs['reviewerID']=nrecs['UserID_index'].map(dict1)
nrecs['ProductID']=nrecs['ProductID_index'].map(dict2)
nrecs=nrecs.sort_values('reviewerID')
nrecs.reset_index(drop=True, inplace=True)
new=nrecs[['reviewerID','ProductID','Rating']]
new['recommendations'] = list(zip(new.ProductID, new.Rating))
res=new[['reviewerID','recommendations']]  
res_new=res['recommendations'].groupby([res.reviewerID]).apply(list).reset_index()
res_new.to_csv(CSV_PATH, index=False)
print(res_new)