#!/usr/bin/env python3

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

with open('gridlock/VERSION.py', 'rt') as f:
    version = f.readlines()[2].strip()

setup(name='gridlock',
      version=version,
      description='Coupled gridding library',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jan Petykiewicz',
      author_email='jan@mpxd.net',
      url='https://mpxd.net/code/jan/gridlock',
      packages=find_packages(),
      package_data={
          'gridlock': ['py.typed'],
      },
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

