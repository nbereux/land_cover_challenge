from setuptools.config import read_configuration
from setuptools import setup, find_packages

setup(
    name="lcc",
    version="1.0.0",
    author="Nicolas BEREUX, Theo DENORME",
    author_email="nicolas.bereux@gmail.com",
    packages=find_packages(),
    install_requires=['python_version >= "3.10.4"'],
)