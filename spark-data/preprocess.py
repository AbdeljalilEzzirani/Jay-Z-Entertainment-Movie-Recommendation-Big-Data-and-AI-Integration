from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, avg

spark = SparkSession.builder \
    .appName("MovieLens Preprocessing") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

# Load datasets from shared volume
movies_df = spark.read.csv("/opt/spark-data/ml-latest-small/movies.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("/opt/spark-data/ml-latest-small/ratings.csv", header=True, inferSchema=True)

movies_df = movies_df.na.drop(subset=["movieId", "title"])
ratings_df = ratings_df.na.drop(subset=["userId", "movieId", "rating"])

movies_df = movies_df.withColumn("movieId", col("movieId").cast("integer"))
ratings_df = ratings_df.withColumn("userId", col("userId").cast("integer")) \
                      .withColumn("movieId", col("movieId").cast("integer")) \
                      .withColumn("rating", col("rating").cast("float"))

movies_df = movies_df.withColumn("genres", split(col("genres"), "\\|"))

avg_ratings_df = ratings_df.groupBy("movieId").agg(avg("rating").alias("avg_rating"))

movies_with_ratings_df = movies_df.join(avg_ratings_df, "movieId", "left")

print("Sample of preprocessed movies with ratings:")
movies_with_ratings_df.show(10)

movies_with_ratings_df.write.mode("overwrite").parquet("/opt/spark-data/processed/movies_with_ratings")

spark.stop()