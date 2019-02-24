#!/usr/bin/env python
import os
from distutils.errors import CCompilerError, DistutilsExecError, DistutilsPlatformError, DistutilsError

from numpy.distutils.core import setup, Extension
from numpy.distutils.command.build_ext import build_ext as old_build_ext
from numpy.distutils.fcompiler import CompilerNotFound


class BuildFailed(Exception):
    pass


def construct_build_ext(build_ext):
    # This class allows extension building to fail.
    # https://stackoverflow.com/questions/41778153/
    ext_errors = (CCompilerError, DistutilsExecError,
                  DistutilsPlatformError, DistutilsError, IOError)
    class WrappedBuildExt(build_ext):
        def run(self):
            try:
                build_ext.run(self)
            except (DistutilsPlatformError, CompilerNotFound) as x:
                raise BuildFailed(x)

        def build_extension(self, ext):
            try:
                build_ext.build_extension(self, ext)
            except ext_errors as x:
                raise BuildFailed(x)
    return WrappedBuildExt


with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()


direct_files = ['direct.pyf', 'DIRect.f', 'DIRserial.f', 'DIRsubrout.f']
direct_paths = [os.path.join('dragonfly', 'utils', 'direct_fortran', x)
                for x in direct_files]
ext1 = Extension(name='dragonfly.utils.direct_fortran.direct',
                 sources=direct_paths)

setup_options = dict(
    name='dragonfly',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/dragonfly/dragonfly/',
    license='MIT',
    packages=['dragonfly', 'dragonfly.apis', 'dragonfly.distributions', 'dragonfly.exd',
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
)

try:
    # Try building the Fortran extension.
    setup(
        ext_modules=[ext1],
        cmdclass={"build_ext": construct_build_ext(old_build_ext)},
        **setup_options
    )
except BuildFailed:
    print("")
    print("*" * 80)
    print("Fortran compilation failed. Falling back on pure Python version.")
    print("*" * 80)
    setup(**setup_options)
