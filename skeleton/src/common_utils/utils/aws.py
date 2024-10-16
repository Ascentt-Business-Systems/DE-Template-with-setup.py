import logging
from typing import Any, Dict
from urllib.parse import urlparse

import boto3

logger = logging.getLogger("common.utils.aws")


class S3Url:
    """
    Parse s3 url into bucket
    >>> s = S3Url("s3://bucket/hello/world")
    >>> s.bucket
    'bucket'
    >>> s.key
    'hello/world'
    >>> s.url
    's3://bucket/hello/world'
    @reference: https://stackoverflow.com/a/42641363
    """

    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        """Get S3 bucket with a given S3 URI"""
        return self._parsed.netloc

    @property
    def key(self):
        """Get S3 key with a given S3 URI"""

        s3_key = ""

        if self._parsed.query:
            s3_key = self._parsed.path.lstrip("/") + "?" + self._parsed.query
        else:
            s3_key = self._parsed.path.lstrip("/")

        return s3_key

    @property
    def url(self):
        """Convert to S3 URL with a given S3 URI"""
        return self._parsed.geturl()


def get_aws_credentials(
    aws_profile: str = "default",
    aws_cred_path: str = None,
    return_pandas_storage_options: bool = False,
) -> Dict:
    """Get the aws credentials from ~/.aws/credentials for the aws profile name

    Params:
        aws_profile (str): AWS credential profile name. Defaults to 'default'.
        aws_cred_path (str): Path to AWS credentials if given. Defaults to None.
        return_pandas_storage_options (bool): Flag if it returns aws credentials or
            pandas storage options. Defaults to False.

    Returns:
        Dict: AWS credential dictionary or dictionary value for pandas storage options
    """
    import configparser
    import os
    import warnings

    # path to the aws credentials file
    path = (
        os.path.join(os.environ["HOME"], ".aws", "credentials")
        if aws_cred_path is None
        else aws_cred_path
    )
    if os.path.isfile(path) is False:
        warnings.warn(f"Cannot locate {path} for aws credentials")
        return {}

    # parse the aws credentials file
    config = configparser.ConfigParser()
    config.read(path)

    # read in the aws_access_key_id and the aws_secret_access_key and the aws_session_token
    # if the profile does not exist, error and exit
    if aws_profile not in config.sections():
        warnings.warn(f"Cannot find profile '{aws_profile}' in {path}")
        return {}

    cred_keys = ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]
    aws_creds = {cred_key: config[aws_profile].get(cred_key) for cred_key in cred_keys}

    # if we don't have both the access and secret key, error and exit
    for cred_key, cred_val in aws_creds.items():
        if cred_val is None:
            raise ValueError(
                f"AWS credential key, {cred_key}, not set in '{aws_profile}' in {path}"
            )

    if return_pandas_storage_options is True:
        pandas_keys = ["key", "secret", "token"]
        lookup = {k: v for k, v in zip(cred_keys, pandas_keys)}
        aws_creds = {lookup[k]: v for k, v in aws_creds.items()}

    return aws_creds


def read_from_s3(s3_path: str, **kwargs: Any) -> bytes:
    """
    Read an s3 object and return the payload
    Args:
        s3_path: Path to s3 object
    Returns: Bytes of the s3 object
    """
    s3_url = S3Url(s3_path)
    s3_client = boto3.resource("s3", **kwargs)
    s3object = s3_client.Object(s3_url.bucket, s3_url.key)
    return s3object.get()["Body"]


def write_to_s3(s3_path, data: Any, *args, **kwargs):
    """Writes arbitrary data to an s3 path

    Args:
        s3_path: s3 desired filepath
        data: arbitrary data

    Returns:
        success of the object being added
    """
    url_obj = S3Url(s3_path)
    s3_client = boto3.resource("s3")

    s3object = s3_client.Object(url_obj.bucket, url_obj.key)
    s3_args = kwargs.pop("s3_additional_kwargs", None)

    logger.info(f"Publishing to {url_obj} with these options {kwargs}")

    if s3_args:
        kwargs.update(s3_args)
    else:
        kwargs = {}

    return s3object.put(Body=data, *args, **kwargs)


def get_secret(secret_name: str, profile_name: str = None, **kwargs) -> dict:
    """
    Get a dictionary from AWS secrets manager
    Args:
        secret_name: Name of the AWS secret
        profile_name: Name of the AWS profile. Defaults to None.

    Returns:
        Dictionary containing the secret values
    """
    session = boto3.Session() if profile_name is None else boto3.Session(profile_name=profile_name)
    client = session.client(service_name="secretsmanager", **kwargs)

    get_secret_value_response = client.get_secret_value(SecretId=secret_name)

    return eval(get_secret_value_response["SecretString"])
