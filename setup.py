from setuptools import setup, find_packages

setup(
    name="yoda_hand",
    version="0.0.1",
    python_requires=">=3.8.0",
    packages=find_packages(exclude=("assets", "DeepSDF", "manotorch", "script")),
)
