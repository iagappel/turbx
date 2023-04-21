#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
import sys

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / 'README.md').read_text()

setup(
    name='turbx',
    version='0.2.3',
    description='Extensible toolkit for analyzing turbulent flow datasets',
    
    long_description=long_description,
    long_description_content_type='text/markdown',
    
    url='https://github.com/iagappel/turbx',
    author='Jason A',
    maintainer='Jason A',
    #author_email='johndoe@gmail.com',
    license='MIT',
    packages=['turbx'],
    #packages=find_packages(exclude=('tests',)),
    install_requires=['mpi4py>=3.1',
                      'numpy>=1.22',
                      'scipy>=1.8',
                      'h5py>=3.6',
                      'matplotlib>=3.6',
                      'scikit-image>=0.19',
                      'psutil>=5.9',
                      'tqdm>=4.64',
                      'cmocean>=2.0',
                      'colorcet>=3.0',
                      'cmasher>=1.6',
                      ],
    
    #setup_requires=['pytest-runner'],
    python_requires='>=3.6',
    #tests_require=['pytest'],
    platforms=['any'],
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    
    keywords=['scientific computing', 'statistics', 'simulation', 'turbulence', 'turbulent flows', 'direct numerical simulation', 'DNS', 'parallel', 'visualization'],
)
