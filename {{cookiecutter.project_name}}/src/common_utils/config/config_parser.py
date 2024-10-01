import json
import logging
import os
import sys
from datetime import datetime, timezone
from glob import glob
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlparse

import cerberus
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

try:
    from common_utils.utils.aws import write_to_s3
except ModuleNotFoundError:
    from src.common_utils.utils.aws import write_to_s3


logger = logging.getLogger("config.config_parser")

ENVIRONMENT_LIST = ["default", "dev", "qa", "prod", "local", "sandbox"]


class SingletonMeta(type):
    _instance = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instance:
            cls._instance[cls] = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instance[cls]


class ConfigParser(metaclass=SingletonMeta):
    """
    Parsing the configuration provided in conf file.
    """

    CONFIG_FOLDER = "/conf/"
    CONFIG_VALIDATION_FOLDER = ".validation/"
    CONFIG_GLOBALS = ["globals", "environment"]
    DEFAULT_DATE = datetime.now().date().strftime("%Y%m%d")

    globals: dict = {}
    parameters: dict = {}
    credentials: dict = {}
    catalog: dict = {}
    logging: dict = {"version": 1}
    # https://docs.python.org/3/library/logging.config.html#dictionary-schema-details

    def __init__(self, path: str = None):
        """Config files path."""
        self.globals = {
            "de_timestamp": self.DEFAULT_DATE,
            "ds_timestamp": self.DEFAULT_DATE,
        }

        # set configuration path
        self.path = path
        self.conf = self._set_conf_path()
        # validation schemas for the yaml files
        self.conf_validation = str(self.conf) + self.CONFIG_VALIDATION_FOLDER

        # Loading the environment from environment variables
        self.env = os.environ.get("ENV", None)
        skip_locals = os.environ.get("skip_locals", "false")
        skip_locals = skip_locals.lower() == "true"

        # Loading the configuration
        self._load_config(skip_locals=skip_locals)

        # Update environment variable with config globals
        self._set_globals()

    def set_env(self, env: str) -> None:
        """
        Set env value
        Args:
            env: Name of environment
        """
        self.env = env
        self.set_runtime_globals({"ENV": env}, skip_globals=False)

    def set_runtime_globals(self, params: dict, skip_globals: bool = True):
        """
        Override the default globals with runtime values and
        re-load the parameters with the new globals
        Args:
            params: Dictionary containing the override values
            skip_globals: Skip reloading of globals
        """
        logger.info(f"Overriding global parameters with {params}")
        self.globals.update(dict({k: v for k, v in params.items()}))
        self._load_config(skip_globals=skip_globals)
        self._set_globals()

    def set_runtime_params(self, params: dict, name: str) -> None:
        """
        Override the default parameters with runtime values
        Args:
            params: Dictionary containing the override values
            name: parameter name
        """
        logger.info(f"Overriding {name} parameters with {params}")

        logger.debug(f"Parameters before update: {self.parameters}")
        config = getattr(self, name)
        config.update(dict({k: v for k, v in params.items()}))
        setattr(self, name, config)
        logger.debug(f"Parameters after update: {self.parameters}")

    def save_run_configuration(self) -> None:
        """
        Persist the configuration to a location defined in the catalog
        """
        root_location = self.catalog.get("ref_configuration_store", {}).get("file_path")
        if root_location is None:
            logger.warning(
                "Catalog value `ref_configuration_store` not found, "
                "configuration will not be tracked"
            )
            return

        is_s3_path = urlparse(root_location).scheme == "s3"
        output_folder_name = self._get_runtime().strftime("%Y%m%d%H%M%S")
        output_folder = Path(root_location) / output_folder_name
        if not is_s3_path and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        logger.info(f"Saving run configuration to {output_folder}")
        configurations_to_persist = [
            ("catalog.yml", self.catalog),
            ("parameters.yml", self.parameters),
        ]

        for file_name, conf in configurations_to_persist:
            contents = yaml.dump(conf, sort_keys=False)
            output_path = os.path.join(output_folder, file_name)
            if urlparse(root_location).scheme == "s3":
                write_to_s3(output_path, contents.encode("UTF-8"))
            else:
                with open(output_path, "w") as _out:
                    _out.write(contents)

    def _get_file_priority(self) -> List:
        """
        Prioritizing the file reading sequence, last one will be highest priority.
        TODO: Make the function dynamic
        Returns:
        A list of file names with "globals" being read first.
        """
        return [
            "globals",
            "environment",
            "catalog",
            "credentials",
            "parameters",
            "logging",
        ]

    def _get_priority(self) -> List:
        """
        Get the priority list for loading the configurations
        Returns:
        A list of folder name with "default" being read first.
        """
        # return the default priority_list in case env variable not found.
        if not self.env:
            return ["default", "local"]

        # Raise error if the environment not found in priority list
        if self.env not in ENVIRONMENT_LIST:
            raise ValueError(
                f"Invalid ENV {self.env}. Please enter one of {ENVIRONMENT_LIST}"
            )

        # If environment is default then just return a
        # default else append the environment on default.
        if self.env == "default":
            return ["default", "local"]
        else:
            return [
                "default",
                f"environment/{self.env}",
                "local",
            ]

    def _get_runtime(self):
        return datetime.now(timezone.utc)

    def _load_config(self, skip_locals: bool = False, skip_globals: bool = False) -> None:
        """Loading the configuration files"""
        files = self._get_file_priority()
        priorities = self._get_priority()

        for f in files:
            if f in self.CONFIG_GLOBALS and skip_globals:
                continue
            for folder in priorities:
                if (folder == "local") and (skip_locals is True):
                    continue
                self._set_config_attr(f, folder)

    def _populate_global_variables(self, file_path: str) -> str:
        """
        Updating the templates with global configuration values.
        Args:
            file_path: File path for yaml file to be rendered.
        Returns:
            Rendered templates
        """

        # Populating the jinja templates
        env = Environment(loader=FileSystemLoader(self.conf), undefined=StrictUndefined)
        rendered = env.get_template(file_path).render(self.globals)

        return rendered.encode("utf-8")

    def _read_schema(self, name: str) -> Dict:
        """
        Read the JSON schema for validation of yml config files
        Args:
            name (str): Name of the file like, globals, catalog etc. Allowed names:
                "globals",
                "catalog",
                "credentials",
                "parameters",
                "logging"
        Returns (Dict):
            A dictionary of schema file for yaml
        """
        if not os.path.isfile(self.conf_validation + f"{name}.json"):
            return {}

        with open(self.conf_validation + f"{name}.json", "r") as f:
            schema = json.loads(f.read())

        return schema

    def _read_yaml(self, file_path: str) -> Dict:
        """
        Reading the passed yaml file.
        Args:
            file_path (str): Path to configuration file.
        Returns (Dict):
            Loaded configuration.
        """
        with open(file_path, "r") as stream:
            try:
                if not os.path.splitext(os.path.basename(file_path))[0] in self.CONFIG_GLOBALS:
                    # Jinja environment root is defined as the conf folder
                    # so the templates need to be referenced from that location
                    folder_name = os.path.basename(os.path.dirname(file_path))
                    file_name = os.path.basename(file_path)
                    stream = self._populate_global_variables(os.path.join(folder_name, file_name))

                config = yaml.safe_load(stream) or {}

                # Validating if configured not to validate
                self._validate_yaml(file_path, config)
                return config
            except yaml.YAMLError:
                raise yaml.YAMLError("Please check your configuration file at" f" {file_path}")
            except cerberus.DocumentError as e:
                raise ValueError(e)

    def _set_conf_path(self) -> Path:
        wd = os.path.abspath(sys.modules[self.__module__].__file__ + "/../../../../")
        wd = wd.rstrip("/")
        conf = self.path if self.path is not None else f"{wd}/{self.CONFIG_FOLDER}"
        return Path(conf)

    def _set_config_attr(self, name: str, folder_path: str) -> None:
        """
        Setting the configuration attributes to get the object access.
        Args:
            name: name of the file ex. globals
            folder_path: folder in which the file resides
        """
        attr_name = "globals" if name in self.CONFIG_GLOBALS else name
        config = getattr(self, attr_name, {})
        for file_path in glob(os.path.join(self.conf, folder_path, f"{name}*yml")):
            parameters = self._read_yaml(file_path)

            for overlapping_key in set(parameters.keys()).intersection(config):
                if name not in self.CONFIG_GLOBALS:
                    logger.debug(
                        f"Key: `{overlapping_key}` overwritten with {parameters[overlapping_key]}"
                    )

            config.update(parameters)
        setattr(self, attr_name, config)

    def _set_globals(self):
        """Override globals as needed by updating the environment"""
        os.environ.update({k: str(v) for k, v in self.globals.items()})

    def _validate_yaml(self, name: str, data: dict) -> dict:
        """
        Validate yaml file data with loaded schema
        Args:
            name (str): name of ythe configuration type
            data (dict): config data that was read
        Returns (dict):
           data: config data that was read in
        Raises (cerberus.DocumentError):
            In case yaml is not valid an error is raised
        """
        schema = self._read_schema(name)
        v = cerberus.Validator({"v": schema})
        is_valid = v.validate({"v": data})
        if not is_valid:
            raise cerberus.DocumentError(f"Invalid yaml format for {name}.yml. \n {v.errors['v']}")
        return data
