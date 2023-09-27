from pathlib import Path
from setuptools import setup

description = ["LightGlue"]

with open(str(Path(__file__).parent / "README.md"), "r", encoding="utf-8") as f:
    readme = f.read()
with open(str(Path(__file__).parent / "requirements.txt"), "r") as f:
    dependencies = f.read().split("\n")

setup(
    name="lightglue",
    version="0.0",
    packages=["lightglue"],
    python_requires=">=3.6",
    install_requires=dependencies,
    author="Philipp Lindenberger, Paul-Edouard Sarlin",
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/cvg/LightGlue/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
