import os

from setuptools import setup

with open('README.rst') as rdm:
    README = rdm.read()

DEPENDENCIES = [
    'loam>=0.3.1',
    'f90nml>=1.0.2',
    'setuptools_scm>=1.15',
]
HEAVY = [
    'numpy>=1.12',
    'scipy>=1.0',
    'pandas>=0.22',
    'h5py>=2.7.1',
    'matplotlib>=3.0',
]

ON_RTD = os.environ.get('READTHEDOCS', None) == 'True'
# heavy dependencies are mocked out on Read the Docs
if not ON_RTD:
    DEPENDENCIES.extend(HEAVY)

setup(
    name='stagpy',
    use_scm_version=True,

    description='Tool for StagYY output files processing',
    long_description=README,

    url='https://github.com/StagPython/StagPy',

    author='Martina Ulvrova, Adrien Morison, Stéphane Labrosse',
    author_email='adrien.morison@gmail.com',

    license='Apache',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    python_requires='>=3.5',
    packages=['stagpy'],
    entry_points={
        'console_scripts': ['stagpy = stagpy.__main__:main']
    },
    include_package_data=True,
    install_requires=DEPENDENCIES,
)
