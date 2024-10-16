"""
Pipeline I/O related functionalities
"""

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Union
from urllib.parse import urlparse

import awswrangler as wr
import boto3
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import s3fs
from pyspark.sql import DataFrame

# Internal imports
try:
    from common_utils.config.config_parser import ConfigParser
    from common_utils.utils.aws import S3Url, read_from_s3, write_to_s3
    from src.common_utils.utils.spark import get_spark_session

except ModuleNotFoundError:
    from src.common_utils.config.config_parser import ConfigParser
    from src.common_utils.utils.aws import S3Url, read_from_s3, write_to_s3
    from src.common_utils.utils.spark import get_spark_session

cfg = ConfigParser()
logger = logging.getLogger("utils.io")


def get_input_info(input_filenames: Dict) -> Dict:
    """
    Get the filepath, format and input params of input data.

    Args:
        input_filenames (dict): The key and value contains entry and filename respectively.

    Returns:
        dict: Returns input filepath, format and input params of input data.
    """
    input_info = {
        entry: {
            "file_path": (
                os.path.join(cfg.catalog[entry]["file_path"].rstrip("/"), filename)
                if filename is not None
                else cfg.catalog[entry]["file_path"]
            ),
            "file_format": cfg.catalog[entry]["file_format"],
            "params": (
                load_data(catalog_entry=entry, suffix=f"{filename}.pkl")
                if filename is not None
                else None
            ),
        }
        for entry, filename in input_filenames.items()
    }

    input_info = {"input_info": input_info}

    return input_info


def get_input_paths(input_filenames: Dict) -> Dict:
    paths_to_input_data = {
        f"path_to_{entry}": {
            k: (
                os.path.join(v.rstrip("/"), filename)
                if (k == "file_path") and (filename is not None)
                else v
            )
            for k, v in cfg.catalog[entry].items()
        }
        for entry, filename in input_filenames.items()
    }
    return paths_to_input_data


def is_exists(catalog_entry: str, **kwargs) -> bool:
    """
    Check if data specified by the catalog entry exists in an S3 bucket.

    Args:
        catalog_entry (str): The key identifying the catalog entry in the configuration.
        **kwargs: Additional keyword arguments to customize the S3 object retrieval.

    Returns:
        bool: True if the data exists, False otherwise.
    """
    s3 = boto3.client("s3")
    data_url = cfg.catalog[catalog_entry]["file_path"]

    kwargs = _merge_kwargs(cfg.catalog[catalog_entry].get("load", {}), kwargs)
    suffix = kwargs.pop("suffix", None)
    data_url = data_url.rstrip("/") if suffix is None else f"{data_url.rstrip('/')}/{suffix}"

    # if data exists in S3
    if data_url.startswith("s3://"):
        s3_url = S3Url(url=data_url)
        logger.debug(json.dumps({"bucket": s3_url.bucket, "key": s3_url.key}, indent=4))
        results = s3.list_objects_v2(Bucket=s3_url.bucket, Prefix=s3_url.key)
        if "Contents" not in results:
            logger.debug(f"The data doesn't exist: {data_url}.")
            return False
    else:
        if os.path.isfile(data_url) is False:
            logger.debug(f"The data doesn't exist: {data_url}.")
            return False

    logger.debug(f"The data exists: {data_url}.")

    return True


def is_require_filename(catalog_entry: str = None, **kwargs: Any) -> bool:
    """Check if the catalog entry requires filename to load data.

    Args:
        catalog_entry (str): Name of catalog entry to check if it requires file name or not.
        kwargs (Any): additional keyword arguments for boto3 client.

    Raises:
        FileNotFoundError: Occurs when catalog entry's data path doesn't exist.

    Returns:
        bool: Flag if it requires filename to load data.
    """
    # data path per given catalog entry
    data_url = kwargs.pop("data_url", None)
    data_url = cfg.catalog[catalog_entry]["file_path"] if data_url is None else data_url

    # s3
    if data_url.startswith("s3://"):
        # list objects per data url
        s3_url = S3Url(url=data_url)
        s3_kwargs = {"Bucket": s3_url.bucket, "Prefix": s3_url.key}
        s3 = boto3.client("s3", **kwargs)
        results = s3.list_objects_v2(**s3_kwargs)

        if "Contents" not in results:
            raise FileNotFoundError(f"Invalid data path: {data_url}.")

        if s3_url.key in [r.get("Key") for r in results["Contents"]]:
            logger.debug(json.dumps({**s3_kwargs, "is_require_suffix": False}, indent=4))
            return False

        logger.debug(json.dumps({**s3_kwargs, "is_require_suffix": True}, indent=4))
        return True

    # local file
    if os.path.exists(data_url) is False:
        raise FileNotFoundError(f"Invalid data path: {data_url}.")

    if os.path.isfile(data_url) is True:
        logger.debug(json.dumps({"file_path": data_url, "is_require_suffix": False}, indent=4))
        return False

    logger.debug(json.dumps({"file_path": data_url, "is_require_suffix": True}, indent=4))
    return True


def load_data(catalog_entry: str, **kwargs) -> Union[DataFrame, pd.DataFrame, List]:
    """
    Loads the data in a dataframe after getting the file information from the catalog.
    Args:
        catalog_entry (str): config entry for fetching the details about the data location/type etc.
        pd_df [Optional] (bool): A boolean indication whether to load a pandas dataframe.
    Returns:
        data: Either pandas or a spark dataframe based on the parameters passed, or a pickled object
    """

    data_format = cfg.catalog[catalog_entry]["file_format"]
    data_url = cfg.catalog[catalog_entry]["file_path"]

    # Merge conf and function input kwargs
    kwargs = _merge_kwargs(cfg.catalog[catalog_entry].get("load", {}), kwargs)
    data_path = data_url + kwargs.pop("suffix", "")

    if (data_format == "pkl") or any(data_path.endswith(f".{ext}") for ext in ["pickle", "pkl"]):
        return _load_pickle_json(data_path, pickle)
    if (data_format == "json") or any(data_path.endswith(f".{ext}") for ext in ["json"]):
        return _load_pickle_json(data_path, json)

    # Check if kwarg contains GE switch or not
    kwarg_run_ge = kwargs.pop("run_ge", False)

    if kwargs.get("pd_df", False):
        del kwargs["pd_df"]
        data = _load_pandas(data_format, data_path, **kwargs)

    elif kwargs.get("pl_df", False):
        del kwargs["pl_df"]

        data = _load_polars(data_format, data_path, **kwargs)

    else:
        if data_format.lower() in ["csv"] and "header" not in kwargs:
            kwargs["header"] = True

        data = _load_spark(data_format, data_path, **kwargs)

    return data


def load_inputs(input_filenames: Dict) -> Dict:
    inputs = {
        entry: (
            load_data(catalog_entry=entry, suffix=filename)
            if filename is not None
            # For development & debugging
            else load_data(catalog_entry=entry)
        )
        for entry, filename in input_filenames.items()
    }
    return inputs


def load_params(param_entry: str) -> Union[Dict, List, str, int, None]:
    """read the parameters from configuration
    Args:
        param_entry (str): Entry of the configuration file
    Returns (Dict, List, str, None):
        Saved parameter in form of Dictionary, List or str.
    """
    return cfg.parameters.get(param_entry, None)


def save_data(
    data: Union[DataFrame, pd.DataFrame, Dict, List, str], catalog_entry: str, **kwargs
) -> Union[None, Any]:
    """
    Saving the data based on the location passed and run greater expectation
    * The passed kwargs will override any arguments passed in the configuration file.
    Args:
        data (spark or Pandas dataframe or dict): dataframe or dictionary to be saved at given url
        catalog_entry (str): config entry for fetching the details about the data location/type etc.
    returns:
        Status of what was successfully saved and run greater expectation
    """

    data_format = cfg.catalog[catalog_entry]["file_format"]
    data_url = cfg.catalog[catalog_entry]["file_path"]
    output_path = data_url + kwargs.pop("suffix", "")

    # Merge conf and function input kwargs
    kwargs = _merge_kwargs(cfg.catalog[catalog_entry].get("save", {}), kwargs)

    # If data is not a dataframe and is a list or dictionary, save it as JSON.
    if isinstance(data, (list, dict, str)) or data_format == "pkl":
        func = json if data_format == "json" else pickle
        return _save_json_pickle(
            data,
            output_path,
            func,
            **kwargs,
        )

    # Check if kwarg contains GE switch or not
    kwarg_run_ge = bool(kwargs.get("run_ge", False))
    kwargs.pop("run_ge", None)

    # Check if kwargs contain tableau related args or not
    to_tableau = kwargs.pop("to_tableau", None)
    if to_tableau:
        logger.info(
            f"data in catalog {catalog_entry} is being saved but not published "
            "- run publish_data to send to tableau"
        )
        kwargs.pop("tableau_env", None)
        kwargs.pop("project_name", None)
        kwargs.pop("project_id", None)
        kwargs.pop("tableau_secret_name", None)
        kwargs.pop("parent_project_name", None)

    if data_format.lower() in ["csv", "excel"]:
        kwargs["header"] = True

    if isinstance(data, pd.DataFrame):
        output_path = output_path.rstrip("/")
        _create_output_folder(output_path)
        return _save_pandas_data(
            data,
            data_format,
            output_path,
            **kwargs,
        )

    if isinstance(data, pl.DataFrame):
        output_path = output_path.rstrip("/")
        _create_output_folder(output_path)
        return _save_polars(
            data,
            data_format,
            output_path,
            **kwargs,
        )

    if isinstance(data, DataFrame):
        return _save_spark(
            data,
            data_format,
            output_path,
            **kwargs,
        )

    raise RuntimeError(f"Invalid data type found for {catalog_entry} of type {type(data)}")


def _create_output_folder(output_path: str) -> None:
    """
    Create local output folder if it doesn't exist
    Args:
        output_path: Output file path
    """
    output_folder = os.path.dirname(output_path)
    if urlparse(output_path).scheme != "s3" and not os.path.exists(output_folder):
        os.makedirs(output_folder)

def _merge_kwargs(conf_kwargs: dict, func_kwargs: dict = None) -> dict:
    """Merge kwarg values from conf and function input, and overwrite conf values
    if exists in function input

    Args:
        conf_kwargs: kwargs parsed from catalog.yml
        func_kwargs: additional kwargs provided from the function input. Defaults to None.

    Returns:
        merged_kwargs: merged kwarg values from both conf and function inputs, with
            latter overwriting the former
    """

    keys_to_exclude = ["file_path", "file_format"]

    # Exclude file path and file format
    merged_kwargs = {k: conf_kwargs[k] for k in conf_kwargs if k not in keys_to_exclude}
    func_kwargs = {k: func_kwargs[k] for k in func_kwargs if k not in keys_to_exclude}

    # Merge both kwargs, overwrite conf value with functional input
    merged_kwargs.update(func_kwargs)

    return merged_kwargs


def _load_pickle_json(data_url: str, func) -> Any:
    """Loads a pickle or json object

    Args:
        data_url: data's location locally or on cloud.
        func: either pickle or json

    Returns:
        Arbitrary pickled or json object
    """

    logger.info(f"Reading data from {data_url}")

    if urlparse(data_url).scheme == "s3":
        return func.loads(read_from_s3(data_url).read())
    else:
        with open(data_url, "rb") as f:
            return func.load(f)


def _load_polars(data_format: str, data_url: str, **kwargs) -> pl.DataFrame:
    """Returns eager read of dataset from s3

    If we are running into performance issues, look into implementing scan_xyz methods
        to lazily read dataframes

    Args:
        data_format: format of the data eg. csv/parquet
        data_url: data's location locally or on cloud

    Raises:
        RuntimeError: If the datatype has not yet been implemenmted

    Returns:
        eager polars dataframe
    """

    logger.info(f"Reading data from {data_url}")

    is_s3 = urlparse(data_url).scheme == "s3"
    if data_format == "parquet":
        fs = s3fs.S3FileSystem() if is_s3 else None
        dataset = pq.ParquetDataset(data_url, filesystem=fs)
        return pl.from_arrow(dataset.read(), **kwargs)
    elif data_format == "csv":
        kwargs["infer_schema_length"] = 10000
        url_or_obj = read_from_s3(data_url) if is_s3 else data_url
        return pl.read_csv(url_or_obj, **kwargs)
    else:
        logger.error(f"Unsupported file type: {data_format} in polars")
        raise RuntimeError(f"polars not implemented for datatype {data_format} on s3")


def _load_pandas(data_format: str, data_url: str, **kwargs) -> pd.DataFrame:
    """
    Loads the data in a form of pandas dataframe
    Args:
        data_format: format of the data eg. csv/parquet
        data_url: data's location locally or on cloud.
    Returns:
        A pandas dataframe with the data
    """

    pandas_functions = {
        "pickle": pd.read_pickle,
        "table": pd.read_table,
        "csv": pd.read_csv,
        "fwf": pd.read_fwf,
        "clipboard": pd.read_clipboard,
        "excel": pd.read_excel,
        "json": pd.read_json,
        "html": pd.read_html,
        "hdf": pd.read_hdf,
        "feather": pd.read_feather,
        "parquet": pd.read_parquet,
        "orc": pd.read_orc,
        "sas": pd.read_sas,
        "spss": pd.read_spss,
        "sql_table": pd.read_sql_table,
        "sql_query": pd.read_sql_query,
        "sql": pd.read_sql,
        "gbq": pd.read_gbq,
        "stata": pd.read_stata,
    }
    try:
        logger.info(f"Reading data from {data_url}")
        return pandas_functions[data_format](data_url, **kwargs)
    except KeyError as err:
        logger.error(f"Unsupported file type: {data_format}")
        raise err


def _load_spark(data_format: str, data_url: str, **kwargs) -> DataFrame:
    """
    Loads a pyspark dataframe
    Args:
        data_format: format of the data
        data_url: data's location
    Returns:
        a Pyspark dataframe after reading the file
    """

    if kwargs.get("spark"):
        spark = kwargs["spark"]
    else:
        spark = get_spark_session()

    logger.info(f"Reading data from {data_url}")
    # Exception for reading an excel as dataframe, hard dependency on pandas
    if data_format == "excel":
        return spark.createDataFrame(_load_pandas(data_format, data_url, **kwargs).astype(str))
    return spark.read.format(data_format).load(data_url, **kwargs)


def _save_polars(data: pl.DataFrame, data_format: str, data_url: str, **kwargs) -> None:
    """
    Saves a polars dataframe
    documentation is sparse around how to write to cloud currently, so just a wrapper for
    pandas save function this means some functions will not be supported (eg avro)

    all I/O functions per API reference as of 4/3/23
    polars_functions = {
        "json": "write_json",
        "parquet": "write_parquet",
        "csv": "write_csv",
        "ipc": "write_ipc", # feather
        "avro": "write_avro",
        "excel": "write_excel"
    }

    Args:
        data: polars dataframe to save
        data_format: format of the data eg csv, parquet
        data_url: url where the data needs to be stored.
    """

    _save_pandas_data(data.to_pandas(), data_format, data_url, **kwargs)


def _save_pandas_data(data: pd.DataFrame, data_format: str, data_url: str, **kwargs) -> None:
    """
    Saving the pandas dataframe
    Args:
        data: a pandas dataframe that needs to be saved
        data_format: format of the data eg csv, parquet
        data_url: url where the data needs to be stored.
    """
    pandas_functions = {
        "pickle": "to_pickle",
        "csv": "to_csv",
        "clipboard": "to_clipboard",
        "excel": "to_excel",
        "json": "to_json",
        "html": "to_html",
        "latex": "to_latex",
        "feather": "to_feather",
        "parquet": "to_parquet",
        "orc": "to_orc",
        "sql": "to_sql",
        "stata": "to_stata",
    }

    try:
        s3_additional_kwargs = kwargs.pop("s3_additional_kwargs", None)
        logger.info(f"Saving the data in {data_url}")
        # This block enables writing dataframes with server-side encryption to s3 (KMS)
        if s3_additional_kwargs:
            getattr(wr.s3, pandas_functions[data_format])(
                df=data, path=data_url, s3_additional_kwargs=s3_additional_kwargs, **kwargs
            )
        else:
            getattr(data, pandas_functions[data_format])(data_url, **kwargs)
    except KeyError as err:
        logger.error(f"Unsupported filetype {data_format}")
        raise err


def _save_json_pickle(data: Dict, data_url: str, func, **kwargs) -> Union[None, Dict]:
    """
    Saving the json on s3 or locally
    Args:
        data: data to be saved
        data_url: url for the data
        func: pickle or json
    Returns:
        status of json save
    """
    logger.info(f"Writing the json file to {data_url}")
    if urlparse(data_url).scheme == "s3":
        return _save_json_pickle_file_on_s3(data_url, data, func, **kwargs)

    if func == json:
        with open(data_url, "w", encoding="utf-8") as f:
            func.dump(data, f)
    else:
        with open(data_url, "wb") as f:
            func.dump(data, f)


def _save_spark(data: DataFrame, data_format: str, data_url: str, **kwargs) -> None:
    """
    Saving the spark dataframe based on catalog information
    Args:
        data: The spark dataframe to be saved
        data_format: format of the data eg csv, parquet
        data_url: url where the data needs to be stored.
        mode (optional): default is overwrite could take any values from spark writer
    Returns:
        save status of the spark write
    """
    mode = kwargs.get("mode", "overwrite")
    df_writer = getattr(data, "write")
    if kwargs.get("partitionBy"):
        df_writer = df_writer.partitionBy(kwargs.get("partitionBy"))

    logger.info(f"Writing the spark dataframe to {data_url}")
    return df_writer.mode(mode).format(data_format).save(data_url, **kwargs)


def _save_json_pickle_file_on_s3(s3_url: str, data: Dict, func, *args, **kwargs) -> Dict:
    """
    Saving the json file to s3
    Args:
        s3_url: s3 url of where the file will be stored
        data: data in dictionary format to be dumped
        func: either json or pickle
    Return:
        A dictionary of output from Boto3 S3.
    """
    data = func.dumps(data)

    if func == json:
        data = data.encode("UTF-8")

    return write_to_s3(s3_url, data, *args, **kwargs)


# def publish_data(data: DataFrame, catalog_entry: str, **kwargs) -> None:
#     """Publishes a dataframe as a hyperfile to a tableau environment

#     This will overwrite any existing hyperfile in the same tableau env with the same name
#     Can pass a prefix if necessary (eg for publishing for different NAMCs)

#     Args:
#         data: arbitrary p
#         catalog_entry: the name of the catalog entry (from catalog.yml) that will be
#             the name of the hyperfile. This should already have all the necessary tableau
#             arguments

#     Returns:
#         None
#     """

#     data = clean_spark_dtypes(data)
#     # Merge conf and function input kwargs
#     kwargs = _merge_kwargs(cfg.catalog[catalog_entry].get("save", {}), kwargs)

#     # Check if kwargs contain tableau related args or not
#     tableau_env = kwargs.pop("tableau_env", None)
#     project_id = kwargs.pop("project_id", None)
#     project_name = kwargs.pop("project_name", None)
#     parent_project_name = kwargs.pop("parent_project_name", None)
#     tableau_secret_name = kwargs.pop("tableau_secret_name", None)

#     # Send to tableau if specified
#     logger.info(f"Using Tableau environment: {tableau_env}")
#     env_name = list(tableau_env.keys())[0]
#     user_pass = get_secret(tableau_secret_name)
#     # Secret stored as a dict of {tableau_username:user, tableau_password:pass}
#     tableau_env[env_name]["username"] = user_pass["tableau_username"]
#     tableau_env[env_name]["password"] = user_pass["tableau_password"]

#     tableau_prefix = cfg.globals.get("tableau_prefix", "")
#     if tableau_prefix:
#         tableau_prefix = tableau_prefix + "_"

#     catalog_entry = f"{tableau_prefix}{catalog_entry}"

#     hyper_extract = HyperDataSet(
#         data, catalog_entry, tableau_env, project_id, project_name, parent_project_name
#     )
#     hyper_extract.save()
