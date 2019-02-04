#!/usr/bin/env python
import os
from numpy.distutils.core import setup, Extension

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()


direct_files = ['direct.pyf', 'DIRect.f', 'DIRserial.f', 'DIRsubrout.f']
direct_paths = [os.path.join('dragonfly', 'utils', 'direct_fortran', x) for x in direct_files]
ext1 = Extension(name='dragonfly.utils.direct_fortran.direct',
                 sources=direct_paths)

setup(
    name='dragonfly',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/dragonfly/dragonfly/',
    license='MIT',
    packages=['dragonfly', 'dragonfly.distributions', 'dragonfly.exd',
              'dragonfly.gp', 'dragonfly.opt', 'dragonfly.parse',
              'dragonfly.sampling', 'dragonfly.utils', 'dragonfly.utils.direct_fortran',
             ],
    scripts=['bin/dragonfly-script.py'],
    install_requires=[
        'future',
        'numpy',
        'scipy',
        'six',
    ],
    classifiers=[
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: MIT License",
    ],
    ext_modules=[ext1],
)

