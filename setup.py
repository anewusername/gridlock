#!/usr/bin/env python

from distutils.core import setup

setup(name='gridlock',
      version='0.1',
      description='Coupled gridding library',
      author='Jan Petykiewicz',
      author_email='anewusername@gmail.com',
      url='https://mpxd.net/gogs/jan/gridlock',
      packages=['gridlock'],
      install_requires=[
            'numpy',
            'float_raster',
      ],
      dependency_links=[
            'git+https://mpxd.net/gogs/jan/float_raster.git@release#egg=float_raster-0.1'
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

