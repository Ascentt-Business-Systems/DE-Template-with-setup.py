"""
Helper functions to set up pipeline and configs
"""

import argparse
import importlib
import logging
import os
import re
import sys
from pathlib import Path
from pkgutil import iter_modules
from typing import Generator, List
from urllib.parse import urlparse

import yaml
from tqdm.auto import tqdm

# Internal imports
try:
    from common_utils.config.config_parser import ConfigParser
    from common_utils.utils._logging import setup_logging_config
    from common_utils.utils.aws import read_from_s3
except ModuleNotFoundError:
    from src.common_utils.config.config_parser import ConfigParser
    from src.common_utils.utils._logging import setup_logging_config
    from src.common_utils.utils.aws import read_from_s3


PARENT_MODULE = "de_module" ## Update the module name as per your requirements


cfg = ConfigParser()
logger = logging.getLogger("utils.helpers")


OPTIONAL_LAYERS = ["de.optional_tests"]
MANDATORY_LAYERS = [
    "de.raw",
    "de.intermediate",
    "de.primary",
    "de.feature",
    "de.model_input",
    "ds.split",
    "ds.models",
    "ds.model_output",
    "rpt.model_output",
    "rpt.dashboard",
    "rpt.publish",
    "rpt.ui",
]
SORTED_LAYERS = MANDATORY_LAYERS + OPTIONAL_LAYERS


def get_working_directory() -> str:
    """Get current working directory"""
    return str(Path(os.path.dirname(os.path.realpath(__file__))).parents[0].parents[0])


def get_root_directory() -> str:
    """Get project root directory"""
    return str(
        Path(os.path.dirname(os.path.realpath(__file__)))
        .parents[0]
        .parents[0]
        .parents[0]
        .parents[0]
    )


def list_submodules(module) -> Generator:
    """List all submodules of a given module"""
    for submodule in iter_modules(module.__path__):
        yield submodule.name


def load_runtime_parameters(file_path: str) -> dict:
    """
    Load a YAML file from s3 or local

    Args:
        file_path: Local or s3 URI
    Returns:
        Dictionary of parameters
    """

    if os.path.exists(file_path):
        with open(file_path, "r") as stream:
            return yaml.safe_load(stream)
    elif urlparse(file_path).scheme == "s3":
        s3_data = read_from_s3(file_path)
        return yaml.safe_load(s3_data)


def run_modules(**kwargs) -> None:
    """
    Run all or selected modules at once in form of pipeline.
    All arguments are optional if nothing is passed all the modules will run,
    based on pipeline in sequence and alphabetically sorted order of pipelines.
    Only one of {module, layers, skip-layers} is parsed.
    Args:
        module [OPTIONAL] (string):[P1] Run a specfic module
        layers [OPTIONAL] (list): [P2] A list of layers from
            ["raw", "intermediate", "primary", "feature", "model_input", "reporting", "publish"]
        skip_layers [OPTIONAL] (list): [P3] Skip any particular layer from layer list
        skip_api [OPTIONAL] (boolean): Skip all the API modules at once
        skip_modules [OPTIONAL] (list): Skip all the modules in the list
    """
    setup_logging_config(cfg)
    logger.info("Successfully updated the logging configuration!")

    if kwargs.get("env"):
        cfg.set_env(kwargs["env"])

    # Update the parameters with runtime overrides
    if kwargs.get("globals"):
        runtime_globals = load_runtime_parameters(kwargs["globals"])
        cfg.set_runtime_globals(runtime_globals)

    if kwargs.get("user"):
        user_name = kwargs.get("user")
        params = {
            "de_user": user_name,
            "ds_user": user_name,
        }
        cfg.set_runtime_globals(params)

    if kwargs.get("parameters"):
        runtime_params = load_runtime_parameters(kwargs["parameters"])
        cfg.set_runtime_params(runtime_params, "parameters")

    if kwargs.get("catalog"):
        runtime_catalog = load_runtime_parameters(kwargs["catalog"])
        cfg.set_runtime_params(runtime_catalog, "catalog")

    logger.info("Saving configuration for this run!")
    cfg.save_run_configuration()

    pipeline = Pipelines(kwargs)

    # Looping through through sorted modules
    for m in tqdm(pipeline.get_selected_modules()):
        logger.info(f"Running module, de_module.{m}")
        sub_module = importlib.import_module(f".{m}", PARENT_MODULE)
        # Add skip modules logic
        if hasattr(sub_module, "run") and callable(getattr(sub_module, "run")):
            sub_module.run()
        else:
            logger.error(f"Layer: {sub_module} does not have a run method")


class MAINArguments:
    """
    Setup the command line arguments for __main__ file.
    """

    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            description="DE Pipeline", # Update the description
            usage="""python <command> [<args>]""",
        )

        self.subparsers = self.parser.add_subparsers(
            title="subcommands",
            dest="command",
            help="Available sub commands",
            metavar="<command>",
        )

        self.optional_args()
        self.pipeline()

        self.args = self.parser.parse_args(sys.argv[1:])

    @staticmethod
    def __module(astring: str) -> str:
        """
        Checks if the string is a valid module of format, `layer.module`
        """
        if not re.match(r"(.+?)\.(.+?)", astring):
            raise argparse.ArgumentTypeError("Module should be of {layer}.{pipeline} format")
        return astring

    def optional_args(self):
        """Parse optional args"""
        self.parser.add_argument(
            "-p",
            "--parameters",
            help="Parameter file to override defaults, this will take precedence over the env argument as well",  # noqa: E501
        )

        self.parser.add_argument(
            "-c",
            "--catalog",
            help="Catalog file to override defaults, this will take precedence over the env argument as well",  # noqa: E501
        )

        self.parser.add_argument(
            "-g",
            "--globals",
            help="Globals file to override defaults, this will take precedence over the env argument as well",  # noqa: E501
        )

        self.parser.add_argument(
            "-e",
            "--env",
            help="Environment to load",
        )

        self.parser.add_argument(
            "-u",
            "--user",
            help="User override for ds_user and de_user, takes precedence over the --globals flag",
        )

    def pipeline(self):
        """Construct pipeline with command line arguments read in"""
        pipeline_parser = self.subparsers.add_parser(
            "pipeline", help="Running the pipeline command"
        )
        group = pipeline_parser.add_argument_group()

        group.add_argument(
            "-l",
            "--layers",
            nargs="+",
            help="selects layer/s to run",
        )

        group.add_argument(
            "--skip-layers",
            nargs="+",
            help="skips particular layer/s to run",
        )

        group.add_argument(
            "--module",
            type=self.__module,
            help="runs a specific module",
        )

        group.add_argument(
            "--modules",
            type=self.__module,
            nargs="+",
            help="runs specific modules in the defined order",
        )

        group.add_argument(
            "--skip-modules",
            type=self.__module,
            nargs="+",
            help="Skips particular modules",
        )


class Pipelines:
    """
    Parse the arguments to get layers or modules.
    """

    def __init__(self, args: dict) -> None:
        self.args = args

    def get_layers(self) -> list:
        """Get the specified layers to run in order"""

        if self.args.get("layers"):
            layers = self.args["layers"]

        elif self.args.get("skip_layers"):
            skip_layers = self.args["skip_layers"]
            layers = [x for x in MANDATORY_LAYERS if x not in skip_layers]

        elif self.args.get("module"):
            module = self.args["module"].split(".")
            pipeline = module[0]
            layer = module[1]
            layers = [f"{pipeline}.{layer}"]

        else:
            layers = MANDATORY_LAYERS.copy()

        layers.sort(key=SORTED_LAYERS.index)

        return layers

    def get_all_modules(self, layers: list) -> list:
        """
        Get all the modules from selected layers.
        Args:
            layers: A list of select layers to be looked at.
        Returns (list):
            A list of all the modules in selected layers.
        """
        all_modules = []
        for layer in layers:
            modules = importlib.import_module(f".{layer}", PARENT_MODULE)
            # Looping through through sorted modules
            for m in sorted(list_submodules(modules)):
                all_modules.append(f"{layer}.{m}")
        return all_modules

    def get_modules(self, modules: list) -> list:
        """
        get the modules based on input modules and configuration settings
        Args:
            modules: a list of modules
        """
        if self.args.get("module"):
            modules = [self.args.get("module")]
        elif self.args.get("modules"):
            modules = self.args.get("modules")

        if self.args.get("skip_modules"):
            modules = [m for m in modules if m not in self.args.get("skip_modules", [])]

        return modules

    def get_selected_modules(self) -> List[str]:
        """
        Get only the selected modules from all the modules
        """
        layers = self.get_layers()
        all_modules = self.get_all_modules(layers)

        return self.get_modules(all_modules)
