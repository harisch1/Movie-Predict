from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col, explode

from pyspark.ml.recommendation import ALS

from pyspark.ml.evaluation import RegressionEvaluator

spark = SparkSession.builder.getOrCreate()

#Reading from the file

ratings_df = spark.read.csv("Rating.csv", header=True, inferSchema=True)

ratings_df = ratings_df.where("rating != -1")

#initialzing recommender
recommender = ALS(maxIter=25, regParam=0.1, userCol="user_id", 
                      itemCol = "teleplay_id", ratingCol = "rating", 
                      coldStartStrategy = "drop")

train, test = ratings_df.randomSplit([0.9, 0.1])

trained = recommender.fit(train)

prediction = trained.transform(test)

prediction.show()

# Create a user for recommending
user_input = ratings_df.select("user_id").where(col("user_id") == 53698)   

user = trained.recommendForUserSubset(user_input, 100000)

user = user.withColumn("rec_exp", explode("recommendations")).select('user_id', 
                        col("rec_exp.teleplay_id"), 
                        col("rec_exp.rating"))

user.show(10, False)

evaluator = RegressionEvaluator(metricName="rmse", 
                                labelCol="rating",
                                predictionCol="prediction")

rmse = evaluator.evaluate(prediction)
print("RMSE = " + str(rmse))

user.coalesce(1).write.csv("18086809d_task2.csv", header=True, mode="overwrite")

