#!/usr/bin/env python3

from setuptools import setup, find_packages
import gridlock

setup(name='gridlock',
      version=gridlock.version,
      description='Coupled gridding library',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/code/jan/gridlock',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'float_raster',
      ],
      extras_require={
          'visualization': ['matplotlib'],
          'visualization-isosurface': [
                'matplotlib',
                'skimage>=0.13',
                'mpl_toolkits',
          ],
      },
      )

