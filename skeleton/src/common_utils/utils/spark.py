import os
from typing import Dict

import pandas as pd
from pyspark import SparkConf
from pyspark.sql import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.types import FloatType


def is_running_dbx() -> bool:
    """Check if the pipeline is currently running via DataBricks"""
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))


def get_spark_session(
    master: str = "local[*]",
    app_name: str = "de-modules",
):
    """Generate spark session, using either existing one from DBX
    or create one locally
    """
    if is_running_dbx():
        spark = SparkSession.builder.getOrCreate()
    else:
        try:
            spark = SparkSession.builder.master(master).appName(app_name).getOrCreate()
        except Exception:
            conf = get_local_spark_config()
            spark = (
                SparkSession.builder.master(master)
                .appName(app_name)
                .config(conf=conf)
                .getOrCreate()
            )

    return spark


def get_local_spark_config(aws_creds: Dict = None) -> SparkConf:
    conf = SparkConf()

    conf.set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.2.0")
    conf.set(
        # https://stackoverflow.com/a/73504175
        "spark.driver.extraJavaOptions",
        (
            "--add-opens=java.base/java.lang=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED "
            "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED "
            "--add-opens=java.base/java.io=ALL-UNNAMED "
            "--add-opens=java.base/java.net=ALL-UNNAMED "
            "--add-opens=java.base/java.nio=ALL-UNNAMED "
            "--add-opens=java.base/java.util=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED "
            "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
            "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED "
            "--add-opens=java.base/sun.security.action=ALL-UNNAMED "
            "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED "
            "--add-opens=java.security.jgss/sun.security.krb5=ALL-UNNAMED"
        ),
    )

    if aws_creds is not None:
        conf.set(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider",
        )
        conf.set("spark.hadoop.fs.s3a.access.key", aws_creds["aws_access_key_id"])
        conf.set("spark.hadoop.fs.s3a.secret.key", aws_creds["aws_secret_access_key"])
        conf.set("spark.hadoop.fs.s3a.session.token", aws_creds["aws_session_token"])

    return conf


def to_pandas_safe(df: DataFrame) -> pd.DataFrame:
    # find all decimal columns in your SparkDF
    decimals_cols = [c for c in df.columns if "Decimal" in str(df.schema[c].dataType)]

    # convert all decimals columns to floats
    for col in decimals_cols:
        df = df.withColumn(col, df[col].cast(FloatType()))

    # Now you can easily convert Spark DF to Pandas DF without decimal errors
    pandas_df = df.toPandas()
    return pandas_df
