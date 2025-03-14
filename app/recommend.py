from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

# Start Spark session
spark = SparkSession.builder \
    .appName("Movie Recommendations") \
    .config("spark.master", "spark://spark-master:7077") \
    .getOrCreate()

# Load movies data
movies_df = spark.read.csv("/opt/spark-data/ml-latest-small/movies.csv", header=True, inferSchema=True)

# Load the saved ALS model
model = ALSModel.load("/opt/spark-data/models/als_model")

# Recommend for a specific user (e.g., userId = 1)
user_id = 1
recommendations = model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["userId"]), 10)

# Explode recommendations to join with movies
from pyspark.sql.functions import explode
recs_expanded = recommendations.select("userId", explode("recommendations").alias("rec")).select("userId", "rec.movieId", "rec.rating")

# Join with movies to get titles
recs_with_titles = recs_expanded.join(movies_df, "movieId").select("userId", "movieId", "title", "rating")

# Show recommendations with titles
recs_with_titles.show(truncate=False)

# Stop Spark
spark.stop()