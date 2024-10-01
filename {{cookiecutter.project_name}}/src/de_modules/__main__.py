from src.common_utils.config.config_parser import ConfigParser

from src.common_utils.utils.helpers import MAINArguments, run_modules

cfg = ConfigParser()


def main():
    agms = MAINArguments()
    run_modules(**vars(agms.args))


if __name__ == "__main__":
    main()
