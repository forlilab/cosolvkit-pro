#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import fnmatch
from setuptools import setup, find_packages


def find_files(directory):
    matches = []

    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*'):
            matches.append(os.path.join(root, filename))

    return matches


setup(name="cosolvkit",
      version='0.5.0',
      description="CosolvKit",
      author="Niccolo Bruciaferri, Jerome Eberhardt",
      author_email="forli@scripps.edu",
      url="https://github.com/forlilab/cosolvkit",
      packages=find_packages(exclude=['docs']),
      include_package_data=True,
      zip_safe=False,
      license="LGPL-2.1",
      keywords=["molecular modeling", "drug design",
                "cosolvent", "MD simulations"],
      classifiers=["Programming Language :: Python",
                   "Operating System :: Unix",
                   "Operating System :: MacOS",
                   "Topic :: Scientific/Engineering"],
      install_requires=[
          "PyYAML>=6.0",
          # Protein cleaning is delegated to autopath's PDBPreprocessor rather than duplicated.
          # NOT declared as a bare "autopath": that name on PyPI is an unrelated 0.0.0 placeholder,
          # and PyPI normalises "AutoPath" to it, so either spelling would install a stranger's
          # package. AutoPath is an internal ForliLab project -- install it directly:
          #     pip install "AutoPath @ git+ssh://git@github.com/forlilab/autopath.git"
          # or, for development,  pip install -e /path/to/autopath
      ],
      entry_points={
          'console_scripts': [
              'create_cosolvent_system=cosolvkit.cli.create_cosolvent_system:main',
              'analyze_cosolvent_simulation=cosolvkit.cli.analyze_cosolvent_simulation:main',
              'visualize_hotspots=cosolvkit.cli.visualize_hotspots:main',
          ]
      }
      
)
