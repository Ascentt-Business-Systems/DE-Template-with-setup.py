"""
Utility functions commonly used throughout different part of the pipeline
"""

import logging
import re
from typing import Callable, Iterable, TypeVar, Union

from pandas import DataFrame as pandasDataFrame
from pandas import Series as pandasSeries

try:
    import polars as pl
    from polars import DataFrame as polarsDataFrame
except Exception:
    polarsDataFrame = TypeVar("polarsDataFrame")
    pl = TypeVar("pl")
    print("Unable to load polars in this runtime")

try:
    from pyspark.sql import Column
    from pyspark.sql import DataFrame as sparkDataFrame
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DataType, DoubleType, StringType

    spark = SparkSession.getActiveSession()

    class py_or_udf:
        """Decorator that allows function to be used as either a regular python function
            or pyspark UDF

        Reference: https://medium.com/@ayplam/developing-pyspark-udfs-d179db0ccc87
        """

        def __init__(self, returnType: DataType = StringType()):
            """Decorator initialization

            Args:
                returnType: set return type for UDF. Defaults to StringType().
            """
            self.spark_udf_type = returnType

        def __call__(self, func: Callable):
            """Define decorator call mechanism to dynamically be a python function or pyspark UDF

            Args:
                func: function to invoke
            """

            def wrapped_func(*args, **kwargs):
                # Register as UDF if any input is a pyspark column
                if any(isinstance(arg, Column) for arg in args) or any(
                    isinstance(kwarg, Column) for kwarg in kwargs.values()
                ):
                    return F.udf(func, self.spark_udf_type)(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            return wrapped_func

except Exception:
    Column = TypeVar("Column")
    sparkDataFrame = TypeVar("sparkDataFrame")
    SparkSession = TypeVar("SparkSession")
    F = TypeVar("F")
    DataType = TypeVar("DataType")
    DoubleType = TypeVar("DoubleType")
    StringType = TypeVar("StringType")
    print("Unable to load PySpark in this runtime")

logger = logging.getLogger("utils.utils")


def clean_column_name(column_name: str) -> str:
    """Refactor column names including lower casing, and replacement
        of non-alpha characters with underscores or words.

    Args:
        column_name: a 'dirty' column name

    Returns:
        column_name: a 'clean' column name
    """
    column_new = column_name.lower().strip()
    column_new = re.sub(r"[ :_\-]+", "_", column_new)
    column_new = re.sub("#", "num", column_new)
    column_new = re.sub("%", "pct", column_new)
    column_new = re.sub("[&+]+", "and", column_new)
    column_new = re.sub("[|,/;]+", "or", column_new)
    column_new = re.sub("[().]+", "", column_new)

    return column_new


def clean_pandas_polars_column_names(
    df: Union[pandasDataFrame, polarsDataFrame]
) -> Union[pandasDataFrame, polarsDataFrame]:
    """Refactor pandas/polars column names including lower casing, and replacement
        of non-alpha characters with underscores or words.

    Args:
        df: a pandas/polars dataframe

    Returns:
        cleaned_df: same dataframe with column names in lowercase and
            non-alpha characters substituted
    """

    column_rename_dict = {col: clean_column_name(col) for col in df.columns}

    if isinstance(df, pandasDataFrame):
        cleaned_df = df.rename(columns=column_rename_dict)
    else:
        cleaned_df = df.rename(column_rename_dict)

    return cleaned_df


def clean_spark_column_names(spark_df: sparkDataFrame) -> sparkDataFrame:
    """Refactor spark column names including lower casing, and replacement
        of non-alpha characters with underscores or words.

    Args:
        spark_df: a pyspark dataframe

    Returns:
        cleaned_df: same dataframe with column names in lowercase and
            non-alpha characters substituted
    """

    cleaned_df = spark_df.toDF(*[clean_column_name(col) for col in spark_df.columns])

    return cleaned_df


def clean_spark_dtypes(df: sparkDataFrame) -> sparkDataFrame:
    """Cleans any datatypes as necessary
    currently used to publish to TableauHyperApi but can add additional cleaning as needed

    - cast DecimalType to DoubleType

    Note that this will fail if trying to cast to incompatible dtypes

    Args:
        df: arbitrary dataframe

    Returns:
        df: dataframe with column types cleaned
    """
    # iterate through all cols and cast to double as needed

    for column in df.columns:
        if "DecimalType" in str(df.schema[column].dataType):
            df = df.withColumn(column, F.col(column).cast(DoubleType()))

    return df


def get_hashed_column(
    df: Union[sparkDataFrame, pandasDataFrame, polarsDataFrame],
    cols: Iterable[str],
    hash_name: str,
    zero_override: bool = False,
) -> Union[sparkDataFrame, pandasDataFrame, polarsDataFrame]:
    """Wrapper for hashed column calculator across different df types. Creates a hash based on
        both column name and contents across desired columns

    Args:
        df: arbitrary df
        cols: which columns (including name) to hash
        hash_name: desired name of hash column
        zero_override: whether to overwrite the hash of columns containing all zero
            (i.e., None choice) with 0. Defaults to False

    Returns:
        df of same type containing a hashed column
    """
    # guarantee order for hashing function
    cols = sorted(cols)

    if isinstance(df, polarsDataFrame):
        return _get_hashed_column_polars(df, cols, hash_name, zero_override)
    if isinstance(df, pandasDataFrame):
        return _get_hashed_column_pandas(df, cols, hash_name, zero_override)
    else:
        return _get_hashed_column_spark(df, cols, hash_name, zero_override)


def _get_hashed_column_polars(
    df: polarsDataFrame, cols: Iterable[str], hash_name: str, zero_override: bool
) -> polarsDataFrame:
    """Calculate a hashed column calculator. Creates a hash based on both column name and contents
        across desired columns

    Args:
        df: arbitrary polars df
        cols: which columns (including name) to hash
        hash_name: desired name of hash column
        zero_override: whether to overwrite the hash of columns containing all zero
            (i.e., None choice) with 0

    Returns:
        df_hash: df of same type containing a hashed column
    """

    # hash based on the value (ie 1, 2, 3) + column name (for distinctness)
    hash_list = []
    for col in cols:
        hash_list.extend([pl.col(col), pl.lit(col)])

    df_hash = df.with_columns(pl.concat_str(hash_list).apply(hash).alias(hash_name))

    if zero_override:
        filter_criteria = pl.all(pl.col(col) == 0 for col in cols)
        df_hash = df_hash.with_columns(
            pl.when(filter_criteria).then(0).otherwise(pl.col(hash_name)).alias(hash_name)
        )

    return df_hash


def _get_hashed_column_spark(
    df: sparkDataFrame, cols: Iterable[str], hash_name: str, zero_override: bool
) -> sparkDataFrame:
    raise NotImplementedError


def _get_hashed_column_pandas(
    df: pandasDataFrame, cols: Iterable[str], hash_name: str, zero_override: bool
) -> pandasDataFrame:
    """Calculate a hashed column calculator. Creates a hash based on both column name and contents
        across desired columns

    Args:
        df: arbitrary pandas df
        cols: which columns (including name) to hash
        hash_name: desired name of hash column
        zero_override: whether to overwrite the hash of columns containing all zero
            (i.e., None choice) with 0

    Returns:
        df of same type containing a hashed column
    """
    hash_list = ""
    for col in cols:
        hash_list += df[col].map(str) + col

    df[hash_name] = hash_list.apply(hash)

    if zero_override:
        condition = _get_pandas_zero_conditions(df, cols)
        df.loc[condition, hash_name] = 0

    return df


def _get_pandas_zero_conditions(df: pandasDataFrame, cols: Iterable[str]) -> pandasSeries:
    """Dynamically creates a filter condition where all columns are 0 for filtering on hash

    Args:
        df: arbitrary pandas df containing all cols
        cols: all cols to filter against

    Returns:
        condition: filterable condition of rows with all 0s across desired columns
    """
    condition = None
    for col in cols:
        cond = df[col] == 0
        if condition is None:
            condition = cond
        else:
            condition = condition & cond
    return condition


def rename_cols_polars(df: polarsDataFrame, rename_dict: dict) -> polarsDataFrame:
    """Renames columns in polars if and only if the original column exists

    Args:
        df: arbitrary polars df
        rename_dict: format original: new

    Returns:
        df with columns renamed
    """
    rename_dict_subset = {key: rename_dict[key] for key in rename_dict.keys() if key in df.columns}
    df = df.rename(rename_dict_subset)

    return df


def clean_level(level_mapping: str) -> str:
    """Extracts the name of the feature

    Args:
        level_mapping: feature + # - eg "(2) powertrain"

    Returns:
         feature portion extracted from level_mapping - eg "powertrain"
    """
    return re.sub(r"\([^)]*\) ", "", level_mapping)


def get_num_value(level_mapping: str) -> int:
    """Extracts the numeric portion from level mapping

    Args:
        level_mapping: feature + # - eg "(2) powertrain"

    Returns:
        numeric portion extracted from level_mapping eg "2"
    """
    if level_mapping == "None":
        return None

    nums = re.search(r"\([^)]*\) ", level_mapping).group()
    return int(re.search(r"\d+", nums).group())


def convert_value(value):
    try:
        return str(int(value))
    except Exception:
        return str(value)
