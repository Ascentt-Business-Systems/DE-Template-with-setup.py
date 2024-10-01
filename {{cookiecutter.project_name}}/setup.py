import os
import sys

from setuptools import find_namespace_packages, setup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
modulepath = os.path.join(ROOT_DIR, "src")
if modulepath not in sys.path:
    sys.path.append(modulepath)



# Read the contents of your requirements.txt file
def read_requirements(filename):
    with open(filename, "r") as f:
        return f.readlines()


# Parse the requirements to include only package lines, ignoring --extra-index-url and
# other directives
def parse_requirements(filename):
    lines = read_requirements(filename)
    requirements = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith("--"):
            requirements.append(line)
    return requirements


setup(
    name="{{ cookiecutter.project_slug }}",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    version="0.1.0",
    author="{{ cookiecutter.author_name }}",
    description="{{ cookiecutter.description }}",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license="OSI Approved :: MIT License",
    platforms=["OS Independent"],
    setup_requires=["pytest-runner", "setuptools_scm"],
    use_scm_version=True,
    include_package_data=True,
    install_requires=requirements,
)