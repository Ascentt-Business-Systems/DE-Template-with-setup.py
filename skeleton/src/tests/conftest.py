import os
import sys
from pathlib import Path

# This conftest file is required because the Python Lib CI script calls:
# run -m pytest instead of the setup.py file
# without this block below the tests won't be able to import from application packages

src_folder = os.path.abspath(Path(__file__).parent.parent)
sys.path.insert(0, src_folder)  # need this for tests to properly import 