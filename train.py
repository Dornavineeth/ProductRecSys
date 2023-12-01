from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer
from pyspark.ml.feature import PCA
from pyspark.sql import functions as F

# 1. Loading the dataset in PySpark
spark = SparkSession.builder.appName("RecommendationSystem").getOrCreate()

# Load the dataset from CSV
data = spark.read.csv("data/data.csv", header=True, inferSchema=True)

# 2. Train a truncated SVD model
# Assuming you have the required columns (e.g., "rating", "reviewerID", "productID")
# We'll use ALS as truncated SVD is not directly available in PySpark, and ALS is commonly used for collaborative filtering
# Convert reviewerID and productID to numeric indices
indexer = StringIndexer(inputCols=["reviewerID", "productID"], outputCols=["userIndex", "itemIndex"])
data = indexer.fit(data).transform(data)

# Split the dataset into training and test sets
(train, test) = data.randomSplit([0.8, 0.2], seed=123)


print("Data Splitting Done!")

# Create ALS model
als = ALS(
    userCol="userIndex",
    itemCol="itemIndex",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    implicitPrefs=False
)

# Train the ALS model on the training data
model = als.fit(train)

# 3. Evaluate its performance on the test set with recall@k
# Make predictions on the test data
predictions = model.transform(test)

# Define a function to calculate recall@k
def recall_at_k(predictions, k):
    top_k_predictions = predictions.withColumn(
        "rank", F.row_number().over(Window.partitionBy("userIndex").orderBy(F.desc("prediction")))
    ).filter(F.col("rank") <= k)
    
    total_user_items = predictions.select("userIndex", "itemIndex").distinct().groupby("userIndex").count()
    correct_predictions = top_k_predictions.join(predictions, on=["userIndex", "itemIndex"], how="inner").count()

    recall = correct_predictions / total_user_items
    return recall

# Calculate recall@k (e.g., k=5)
k_value = 5
recall_at_k_value = recall_at_k(predictions, k_value)
print(f"Recall@{k_value}: {recall_at_k_value}")

# 4. Evaluation of system/compute while training model
# Depending on your specific requirements, you may want to monitor metrics during the training process
# Unfortunately, PySpark's ALS does not provide real-time metrics during training
# You may consider logging training progress using custom code if needed

# Stop the Spark session
spark.stop()