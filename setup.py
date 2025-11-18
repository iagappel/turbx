#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import sys

## read contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='turbx',
    version='0.5.2',
    description='Extensible toolkit for analyzing turbulent flow datasets',
    
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    url='https://github.com/iagappel/turbx',
    author='Jason A',
    maintainer='Jason A',
    license='MIT',
    
    packages=find_packages(include=['turbx', 'turbx.*']),
    install_requires=[
                'mpi4py>=4.0',
                'numpy>=2.0',
                'scipy>=1.14',
                'h5py>=3.10',
                'matplotlib>=3.9',
                'tqdm>=4.66',
                ],
    
    python_requires='>=3.11',
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Database',
        'Topic :: System :: Distributed Computing',
    ],
    
    keywords=['scientific computing', 'post-processing', 'statistics', 'simulation', 'turbulence',],
)
