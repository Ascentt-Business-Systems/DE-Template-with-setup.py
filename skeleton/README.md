# {{ cookiecutter.project_name }}

## Local Dev Environment Setup

### Prerequisites

- Python environments (3.9, 3.10, 3.11, etc.)
  - [Conda configuration](https://toyota.atlassian.net/wiki/spaces/SDS/pages/503650976/Python+Local+Dev+Environment#Conda)
- [Poetry](https://toyota.atlassian.net/wiki/spaces/SDS/pages/503650976/Python+Local+Dev+Environment#Poetry)
  - Update poetry's environment to use a desired Python environment (e.g., conda)
    ```shell
    poetry env use /path/to/python/env/bin/python    
    ```
  - Update poetry's configuration to add supplemental source of `jfrog-tmna` with environment variables, `JFROG_EMAIL` and `JFROG_API_KEY`. Use the make command, `setup-local-dev`, in `Makefile` to run this properly with `.env`.
    ```shell
    poetry config http-basic.jfrog-tmna ${JFROG_EMAIL} ${JFROG_API_KEY}
    ```
    - This registers JFrog source in poetry configuration.
      ```shell
      poetry config --list
      ```
    - Troubleshoot
      - [Disabling Keyring](https://pypi.org/project/keyring/)
        - If you experience an error of not being able to store the credential in the key ring, you may disable it by setting `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` in the environment.
        - This will save your JFrog credentials in `/path/to/pypoetry/auth.toml`.
- [GNU Make](https://toyota.atlassian.net/wiki/x/YR9aIQ)
- `pre-commit`
  - Installation
    - Using pip (Python): `pip install pre-commit`
      - We follow this instruction by adding it in `pyproject.toml`. It's your choice though.
    - Using Homebrew (macOS): `brew install pre-commit`
    - Using Conda (via conda-forge): `conda install -c conda-forge pre-commit`
  - Run `pre-commit install` in the root directory of the local repository
    - You must be able to see `{REPO_ROOT_DIR}/.git/hooks/pre-commit`.
- `JFrog API Key`
  - We could promote code base using GitHub release ([tutorial](https://chofer.cloud.toyota.com/docs/default/Component/developer-platform-guides/building-and-deploying/xlr-promote-ci-builds/#promote-using-github-releases)).
  - Get API key per [JFrog local python setup](https://chofer.cloud.toyota.com/docs/default/component/developer-platform-guides/building-and-deploying/local-python-setup/)
  - We will use this key in `.env`. See the following section.
- `Latexmk`
  - A Perl script which you just have to run once and it does everything else for you … completely automagically.
  - Installation
    - Mac OS X: MacTeX (https://sourabhbajaj.com/mac-setup/LaTeX/)
    - Ubuntu on WSL:
        - https://linuxconfig.org/how-to-install-latex-on-ubuntu-20-04-focal-fossa-linux
          ```shell
          sudo apt install texlive-latex-extra
          ```
        - https://zoomadmin.com/HowToInstall/UbuntuPackage/latexmk
          ```shell
          sudo apt-get update -y
          sudo apt-get install -y latexmk
          ```

### Project Configuration

- Environment variables
  - Set the following environment variables in `.env`
    ```
    JFROG_API_KEY=<YOUR_JFROG_API_KEY>
    JFROG_EMAIL=<YOUR_JFROG_EMAIL>
    ```
- Repo configuration
  - `.github`
    - `pull_request_template.md`: Pull request template
    - `workflows`: Placeholder for CI/CD configuration per TMNA Github Actions
- Pre-commit hooks    
  - `.pre-commit-config.yaml`: Configuration file to manage pre-commit hooks
  - Prerequisites
    - Run `pre-commit install` once you clone the repository on your local machine and install all dependencies via `pyproject.toml`.
  - Local hooks
    - `auto-update-requirements-txt`: Update `requirements.txt` and `dev_requirements.txt`
      - Note that it runs based on the existing `poetry.lock`. So make sure `poetry.lock` is up-to-date.
    - `linting`: Run `isort`, `black`, `flake8`, `pylint`.
      - You could find their configurations in `pyproject.toml`, `.flake8`, and `.pylintrc`.
    - `small-tests`: Run small-sized tests.
      - You may control it to fail if coverage is under the specified threhold of `PYTEST_COV_MIN` in `Makefile`.
    - **NOTE**
      - `linting` and `small-tests` require dependencies, which must exist in your local python environment. Pay attention to `language: python` of each hook. Otherwise, git will throw an error due to the missing required dependencies. These dependencies must be specified here, not in `Makefile`. `pre-commit` recognizes python in your local environment without recognizing all the installed dependencies. For more details, you can refer to `.git/hooks/pre-commit`.
- Code style configurations
  - `.flake8`: Configuration file for `flake8`, a Python code style guide enforcement utility.
  - `.pylintrc`: Configuration file for `pylint`, a static code analyzer tool for Python.
- Repository configuration
  - `.gitignore`: List patterns of files/paths to ignore for git commits
  - `Makefile`: Placeholder to organize all local dev commands to reuse in a consistent manner
    - For code quality and formatting,
      - `isort`, `black`, `flake8`, `pylint`, `clean`, `format`
    - For dependencies,
      - `setup-local-dev`: One time configuration with `JFROG_EMAIL` and `JFROG_API_KEY` in `.env`.
      - `requirements`: Run this frequently, whenever opening a new PR at least.
    - For [documentation](https://toyota.atlassian.net/wiki/x/QAJhHw),
      - `init-docs`: Initialize `/docs` for documentation
        - Update `/docs/conf.py` to use Sphinx template after running `init-docs`
      - `docs`: Build documentation
        - [Reference](https://www.sphinx-doc.org/en/master/_downloads/1db87291c47cdf2a82cc635794bf6c44/example_google.py) for docstring to prep for documentation of code base
      - `docs-pdf`: Generate a PDF file for the documentation
      - `serve-docs`: Run Sphinx 
    - For testing,
      - `test-all`, `test-large`, `test-medium`, `test-small`
      - Check out this blog for [Test Sizes](https://testing.googleblog.com/2010/12/test-sizes.html).
  - `pyproject.toml`: Dependency management and tool configuration
    - [tool.poetry]
      - Make sure `name` starts with `tmna` for CI/CD. And specify the code base root folder under `src` in `packages`, e.g.,
        ```yaml
        [tool.poetry]
        name = "tmna-{YOUR_REPO_NAME}"
        packages = [{include = "<CODE_BASE_ROOT_DIR_NAME>", from="src"}]
        ```
    - [[tool.poetry.source]]: Primary and supplemental dependency sources
        ```yaml
        [[tool.poetry.source]]
        name = "PyPI"
        priority = "primary"


        [[tool.poetry.source]]
        name = "jfrog-tmna"
        url = "https://artifactory.tmna-devops.com/artifactory/api/pypi/pypi-dev/simple"
        priority = "supplemental"
        ```
    - [tool.poetry.dependencies]: Main dependencies will be added here. Keep packages required by code base only.
    - [tool.poetry.group.dev.dependencies]: `dev` dependencies will be added.
    - [tool.poetry.group.docs.dependencies]: Keep dependencies for Sphix-based technical documentation.
    - [tool.black]: Configuration for `black`, which is a python code formatter.
    - [tool.isort]: Configuration for `isort`, which is a utility tool that supports `import` sorting, section separation, etc.
    - [pytest]: Configuration for `pytest`
