from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = '0.1'
REQUIRED_PACKAGES = []
DEPENDENCY_LINKS = []

setuptools.setup(
    name='UniformAugment',
    version=_VERSION,
    description='Unofficial PyTorch Reimplementation of UniformAugment',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://github.com/tgilewicz/uniformaugment',
    license='MIT License',
    package_dir={},
    packages=setuptools.find_packages(exclude=['tests']),
)
