#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='gridlock',
      version='0.2',
      description='Coupled gridding library',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/gridlock',
      packages=find_packages(),
      install_requires=[
            'numpy',
            'float_raster',
      ],
      extras_require={
          'visualization': ['matplotlib'],
          'visualization-isosurface': [
                'matplotlib',
                'skimage',
                'mpl_toolkits',
          ],
      },
      )

