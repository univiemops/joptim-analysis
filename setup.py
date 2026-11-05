from setuptools import setup, find_packages

setup(
    name='joptim',
    version='0.2',
    description='Source code for JOptim analysis',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)
