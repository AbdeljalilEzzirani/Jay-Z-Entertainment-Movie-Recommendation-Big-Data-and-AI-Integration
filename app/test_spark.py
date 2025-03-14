import os
from pyspark.sql import SparkSession
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

os.environ["SPARK_WORKER_CORES"] = "1"
os.environ["SPARK_WORKER_MEMORY"] = "1g"
os.environ["SPARK_MASTER"] = "spark://spark-master:7077"

logger.debug("Starting Spark session...")
spark = SparkSession.builder \
    .appName("Test Spark") \
    .master("spark://spark-master:7077") \
    .config("spark.executor.memory", "1g") \
    .config("spark.executor.cores", "1") \
    .config("spark.default.parallelism", "2") \
    .config("spark.submit.deployMode", "client") \
    .config("spark.task.maxFailures", "10") \
    .config("spark.driver.host", "0.0.0.0") \
    .config("spark.port.maxRetries", "50") \
    .config("spark.worker.timeout", "600") \
    .config("spark.dynamicAllocation.enabled", "false") \
    .getOrCreate()

logger.debug("Spark session started!")
sc = spark.sparkContext
logger.debug(f"Worker count: {sc._jsc.sc().getExecutorMemoryStatus().size()}")

try:
    logger.debug("Loading movies.csv...")
    df = spark.read.csv("/opt/spark-data/ml-latest-small/movies.csv", header=True, inferSchema=True)
    logger.debug("Showing first 5 rows...")
    df.show(5)
except Exception as e:
    logger.error(f"Error: {str(e)}")
finally:
    logger.debug("Stopping Spark session...")
    spark.stop()