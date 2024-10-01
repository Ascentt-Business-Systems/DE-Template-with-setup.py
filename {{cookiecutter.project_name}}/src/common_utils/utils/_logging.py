import json
import logging
import logging.config
from typing import Any

import numpy as np

logger = logging.getLogger(f"common.utils.{__name__}")


def setup_logging_config(cfg):
    logging.config.dictConfig(cfg.logging)
    try:
        logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    except Exception:
        pass


def log_array_shapes(**kwargs: Any):
    invalid_args = {
        k: arg for k, arg in kwargs.items() if isinstance(arg, (np.ndarray, np.generic)) is False
    }
    if len(invalid_args) > 0:
        raise ValueError(f"{invalid_args} are not numpy arrays.")
    logger.debug(json.dumps({k: v.shape for k, v in kwargs.items()}))
