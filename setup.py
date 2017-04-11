# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as rdm:
    README = rdm.read()

setup(
    name='stagpy',
    use_scm_version=True,

    description='Tool for StagYY output files processing',
    long_description=README,

    url='https://github.com/StagPython/StagPy',

    author='Martina Ulvrova, Adrien Morison, Stéphane Labrosse',
    author_email='adrien.morison@gmail.com',

    license='GPLv2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ],

    packages = ['stagpy'],
    entry_points = {
        'console_scripts': ['stagpy = stagpy.__main__:main']
        },
    setup_requires=['setuptools_scm'],
    install_requires = [
        'numpy',
        'scipy',
        'f90nml',
        'pandas',
        'matplotlib',
        'seaborn',
        'argcomplete',
        'setuptools_scm',
        ],
)
