from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.sql.functions import explode
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("Movie Recommendations API") \
        .config("spark.master", "spark://spark-master:7077") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.default.parallelism", "8") \
        .getOrCreate()

    logger.info("Loading movies.csv...")
    movies_df = spark.read.csv("/opt/spark-data/ml-latest-small/movies.csv", header=True, inferSchema=True)
    logger.info("Loading ALS model...")
    model = ALSModel.load("/opt/spark-data/models/als_model")
    logger.info("Startup complete!")
except Exception as e:
    logger.error(f"Startup failed: {str(e)}")
    raise

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('userId', type=int)
    if not user_id:
        return jsonify({"error": "Please provide a userId"}), 400

    try:
        logger.info(f"Generating recommendations for user {user_id}...")
        recommendations = model.recommendForUserSubset(spark.createDataFrame([(user_id,)], ["userId"]), 10)
        recs_expanded = recommendations.select("userId", explode("recommendations").alias("rec")) \
                                       .select("userId", "rec.movieId", "rec.rating")
        recs_with_titles = recs_expanded.join(movies_df, "movieId") \
                                        .select("userId", "movieId", "title", "rating") \
                                        .collect()

        recs_list = [{"movieId": row.movieId, "title": row.title, "rating": row.rating} for row in recs_with_titles]
        logger.info(f"Recommendations generated for user {user_id}")
        return jsonify({"userId": user_id, "recommendations": recs_list})
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)