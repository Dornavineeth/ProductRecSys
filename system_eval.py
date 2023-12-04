import os
import time
import shutil
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from argparse import ArgumentParser
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.functions import col, collect_list
from pyspark.ml.evaluation import RegressionEvaluator

def parse_args():
    parser = ArgumentParser(description='Recommendation System')
    parser.add_argument('--memory', type=str, default=4)
    parser.add_argument('--num_threads', type=str, default='1')
    parser.add_argument('--model_dir', type=str, default='output')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--csv_name', type=str, default='predictions.csv')
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--max_iter', type=int, default=10)
    parser.add_argument('--reg_param', type=float, default=0.5)
    return parser.parse_known_args()[0]


def train(args):
    memory_limit = args.memory
    num_threads = args.num_threads
    model_dir = args.model_dir
    model_path = os.path.join(model_dir, f"als_mem_{memory_limit}_thr_{num_threads}")
    csv_file_path = os.path.join(model_dir, args.csv_name)
    data_path =  args.data_path
    rank = args.rank
    max_iter = args.max_iter
    reg_param = args.reg_param

    spark = SparkSession.builder \
    .master(f'local[{num_threads}]') \
    .config("spark.driver.memory", f"{memory_limit}g") \
    .appName('RecommendationSystem') \
    .getOrCreate()

    # Load the dataset from CSV
    data = spark.read.csv(data_path , header=True, inferSchema=True)
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
    als=ALS(maxIter=max_iter,regParam=reg_param,rank=rank,userCol="reviewerID_index",itemCol="productID_index",ratingCol="rating",coldStartStrategy="drop",nonnegative=True)
    model=als.fit(training)
    print("Total Time : ", time.time()-start_time)

    # Save the model
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    model.save(model_path)

    # Evaluate the model for RMSE
    evaluator=RegressionEvaluator(metricName="rmse",labelCol="rating",predictionCol="prediction")
    predictions=model.transform(test)
    print("Prediction Size: ", predictions.count())
    
    rmse=evaluator.evaluate(predictions)
    print("RMSE="+str(rmse))
    # predictions.show()

    # Convert the predictions DataFrame to include all predictions per user
    # Generate top-k recommendations for each user
    userRecs = model.recommendForAllUsers(50)  # Top-50 recommendations for each user

    # Prepare the input for RankingMetrics
    user_ground_truth = test.groupby('reviewerID_index').agg(collect_list('productID_index').alias('ground_truth_items'))
    user_train_items = training.groupby('reviewerID_index').agg(collect_list('productID_index').alias('train_items'))

    # Join the recommendations and ground truth data on the user ID
    user_eval = userRecs.join(user_ground_truth, on='reviewerID_index').join(user_train_items, on='reviewerID_index') \
        .select('reviewerID_index', 'recommendations.productID_index', 'ground_truth_items', 'train_items', 'recommendations.rating')
    user_eval = user_eval.toPandas()
    user_eval['productID_filtered'] = user_eval.apply(lambda x:[b for (b,z) in zip(x.productID_index, x.rating) if b not in x.train_items], axis=1)
    user_eval['rating_filtered'] = user_eval.apply(lambda x:[z for (b,z) in zip(x.productID_index, x.rating) if b not in x.train_items], axis=1)
    def score(predicted, actual, metric):
        """
        Parameters
        ----------
        predicted : List
            List of predicted apps.
        actual : List
            List of masked apps.
        metric : 'precision' or 'ndcg'
            A valid metric for recommendation.
        Raises
        -----
        Returns
        -------
        m : float
            score.
        """
        valid_metrics = ['precision', 'ndcg', 'recall@5','recall@10', 'recall@50']
        if metric not in valid_metrics:
            raise Exception(f"Choose one valid baseline in the list: {valid_metrics}")
        if metric == 'precision':
            m = np.mean([float(len(set(predicted[:k]) 
                                               & set(actual))) / float(k) 
                                     for k in range(1,len(actual)+1)])
        if metric == 'recall@10':
            m = len(set(predicted[:10]) & set(actual))/len(actual)
        if metric == 'recall@5':
            m = len(set(predicted[:5]) & set(actual))/len(actual)
        if metric == 'recall@50':
            m = len(set(predicted[:50]) & set(actual))/len(actual)
        if metric == 'ndcg':
            v = [1 if i in actual else 0 for i in predicted]
            v_2 = [1 for i in actual]
            dcg = sum([(2**i-1)/np.log(k+2,2) for (k,i) in enumerate(v)])
            idcg = sum([(2**i-1)/np.log(k+2,2) for (k,i) in enumerate(v_2)])
            m = dcg/idcg
        return m
    user_eval['precision'] = user_eval.apply(lambda x: score(x.productID_filtered, x.ground_truth_items, 'precision'), axis=1)
    # user_eval['NDCG'] = user_eval.apply(lambda x: score(x.productID_filtered, x.ground_truth_items, 'ndcg'), axis=1)
    user_eval['recall_5'] = user_eval.apply(lambda x: score(x.productID_filtered, x.ground_truth_items, 'recall@5'), axis=1)
    user_eval['recall_10'] = user_eval.apply(lambda x: score(x.productID_filtered, x.ground_truth_items, 'recall@10'), axis=1)
    user_eval['recall_50'] = user_eval.apply(lambda x: score(x.productID_filtered, x.ground_truth_items, 'recall@50'), axis=1)

    MAP = user_eval.precision.mean()
    # avg_NDCG = user_eval.NDCG.mean()
    recall_5 = user_eval.recall_5.mean()
    recall_10 = user_eval.recall_10.mean()
    recall_50 = user_eval.recall_50.mean()
    print("MAP="+str(MAP))
    # print("NDCG="+str(avg_NDCG))
    print("Recall@5="+str(recall_5))
    print("Recall@10="+str(recall_10))
    print("Recall@50="+str(recall_50))

if __name__ == '__main__':
    args = parse_args()
    print(args)
    train(args)






# # Save the predictions
# recs=model.recommendForAllUsers(1).toPandas()
# nrecs=recs.recommendations.apply(pd.Series) \
#             .merge(recs, right_index = True, left_index = True) \
#             .drop(["recommendations"], axis = 1) \
#             .melt(id_vars = ['reviewerID_index'], value_name = "recommendation") \
#             .drop("variable", axis = 1) \
#             .dropna() 
# nrecs=nrecs.sort_values('reviewerID_index')
# nrecs=pd.concat([nrecs['recommendation'].apply(pd.Series), nrecs['reviewerID_index']], axis = 1)
# nrecs.columns = [
        
#         'ProductID_index',
#         'Rating',
#         'UserID_index'
       
#      ]
# md=transformed.select(transformed['reviewerID'],transformed['reviewerID_index'],transformed['ProductID'],transformed['ProductID_index'])
# md=md.toPandas()
# dict1 =dict(zip(md['reviewerID_index'],md['reviewerID']))
# dict2=dict(zip(md['ProductID_index'],md['ProductID']))
# nrecs['reviewerID']=nrecs['UserID_index'].map(dict1)
# nrecs['ProductID']=nrecs['ProductID_index'].map(dict2)
# nrecs=nrecs.sort_values('reviewerID')
# nrecs.reset_index(drop=True, inplace=True)
# new=nrecs[['reviewerID','ProductID','Rating']]
# new['recommendations'] = list(zip(new.ProductID, new.Rating))
# res=new[['reviewerID','recommendations']]  
# res_new=res['recommendations'].groupby([res.reviewerID]).apply(list).reset_index()
# res_new.to_csv(CSV_PATH, index=False)
# print(res_new)