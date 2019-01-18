#!/usr/bin/env python

# http://stackoverflow.com/questions/9810603/adding-install-requires-to-setup-py-when-making-a-python-package
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md', 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='dragonfly',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/dragonfly/dragonfly/',
    license='MIT',
    packages=['dragonfly'],
    scripts=['bin/dragonfly.py'],
    install_requires=[
        'future',
    ],
    classifiers=[
                    "Intended Audience :: Developers",
                    "Intended Audience :: Science/Research",
                    "License :: OSI Approved :: MIT License",
    ]
)

