#!/usr/bin/env python3

from setuptools import setup, find_packages
import gridlock

with open('README.md', 'r') as f:
    long_description = f.read()

setup(name='gridlock',
      version=gridlock.version,
      description='Coupled gridding library',
      long_description=long_description,
      long_description_content_type='text/markdown',
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
      classifiers=[
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: GNU Affero General Public License v3',
            'Topic :: Multimedia :: Graphics :: 3D Rendering',
            'Topic :: Scientific/Engineering :: Electronic Design Automation (EDA)',
            'Topic :: Scientific/Engineering :: Physics',
            'Topic :: Scientific/Engineering :: Visualization',
      ],
      )

