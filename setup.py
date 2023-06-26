from pathlib import Path
from setuptools import setup

description = ['LightGlue']

with open(str(Path(__file__).parent / 'README.md'), 'r', encoding='utf-8') as f:
    readme = f.read()

extra_dependencies = ['torch', 'kornia', 'numpy', 'einops']

setup(
    name='lightglue',
    version='0.0',
    packages=['lightglue'],
    python_requires='>=3.6',
    extras_require={'extra': extra_dependencies},
    author='Philipp Lindenberger, Paul-Edouard Sarlin',
    description=description,
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/cvg/LightGlue/',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
