import os

from setuptools import setup

with open('README.rst') as rdm:
    README = rdm.read()

DEPENDENCIES = [
    'numpy>=1.12',
    'scipy>=1.0',
    'pandas>=0.22',
    'h5py>=2.7.1',
    'matplotlib>=2.0',
    'seaborn>=0.8.1',
    'loam>=0.3.1',
    'f90nml>=1.0.2',
    'setuptools_scm>=1.15',
]

ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'
if ON_RTD:  # heavy dependencies are mocked out
    DEPENDENCIES = DEPENDENCIES[6:]

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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    packages=['stagpy'],
    entry_points={
        'console_scripts': ['stagpy = stagpy.__main__:main']
    },
    install_requires=DEPENDENCIES,
)
