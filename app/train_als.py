from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MovieLens ALS Training") \
    .config("spark.master", "spark://spark-master:7077") \
    .getOrCreate()

# Load preprocessed ratings data (assuming ratings.csv is still needed for ALS)
ratings_df = spark.read.csv("/opt/spark-data/ml-latest-small/ratings.csv", header=True, inferSchema=True)

# Prepare data for ALS (userId, movieId, rating)
als_data = ratings_df.select("userId", "movieId", "rating")

# Define ALS model
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True
)

# Train the model
model = als.fit(als_data)

# Generate top 10 movie recommendations for each user
user_recs = model.recommendForAllUsers(10)

# Show sample recommendations
user_recs.select("userId", "recommendations.movieId", "recommendations.rating").show(5, truncate=False)

# Save the model
model.save("/opt/spark-data/models/als_model")

# Stop Spark session
spark.stop()