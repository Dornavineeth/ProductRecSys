# Product Recommendation System using PySpark.

Leveraging a subset of Amazon’s vast dataset, this project develops a recommendation system to enhance shopping experiences by predicting and personalizing product suggestions, aiming to improve user satisfaction and drive sales. Powered by a collaborative filtering algorithm and based on a PySpark system, we aim to design and implement a robust and efficient system. We aim not only to create an effective recommendation model but also to meticulously analyze its performance and scalability. Thus, we address the critical need for personalized product recommendations in the modern e-commerce landscape.


# Installation

We have two environments because it is challenging to run different softwares in one environment.

**_NOTE:_** Assumes open-sdk-java 11.0 is already installed , and conda env is setted up

```
# Install environment for PySpark
conda create --name pyspark python=3.7
conda activate pyspark
pip install -r requirements_1.txt

# Install environment for baslines
conda create --name baseline
conda activate baseline
pip install -r requirements_2.txt
```

## Preprocessing Data

**_NOTE:_** is not available in the source code. please find the relevant data [here](https://drive.google.com/drive/folders/1ePzmaS7iGE1aFoyiQ9WIq6dGOb3K6fWZ?usp=sharing)

[1] Download Data : [download.ipynb](prepocess/download.ipynb)

[2] Data Cleaning & EDA : [EDA.ipynb](prepocess/EDA.ipynb)


## ALS Hyper parameter Tuning

```
bash scripts/model_tune.sh
```

Please find the results in result folder

## ALS Training & Evaluation

Evalute how rank effect the performance and how multiple threads can speed up the process
```
bash scripts/system_eval.sh
```

Please find the results in result folder.

**_NOTE:_** The model is trained on 24 cores system with memory as 500G memory.

## Baselines

### SVD 

```
basch scripts/svd.sh
```

### User based KNN 

```
basch scripts/user_knn.sh
```