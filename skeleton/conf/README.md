# What is is this for?

This folder contains all the configuration files for our repository.

## Local configuration
The local folder should be used for configuration that is either user-specific (e.g. IDE configuration) or protected (e.g. security keys).

## Default configuration
The default folder is for shared configuration, such as non-sensitive and project-related configuration that may be shared across team members.

WARNING: Please do not put access credentials in the base configuration folder.

## Intructions
We use, jinja templates for loading the configuration files and applying global variables / environment variables to them.

For more information:
### Jinja
- https://ansible-arista-howto.readthedocs.io/en/latest/JINJA_YAML_STRUCTURES.html
- https://medium.com/@luongvinhthao/generate-yaml-file-with-python-and-jinja2-9474f4762b0d
- https://jinja.palletsprojects.com/en/3.1.x/

### Yaml
- https://docs.ansible.com/ansible/latest/reference_appendices/YAMLSyntax.html