from setuptools import find_packages, setup

setup(
    name="sting",
    packages=find_packages(where=".", include="sting*", exclude="test"),
    version="0.1.1",
    author="CSDS 440 TAs",
    description="Library containing boilerplate code for ML applications. Written for CSDS 440",
    url="https://github.com/cwru-all/sting",
    install_requires=["pandas"])
