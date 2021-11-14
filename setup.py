from pathlib import Path
from setuptools import setup


setup(
    name='stagpy',

    description='Tool for StagYY output files processing',
    long_description=Path("README.rst").read_text(),

    url='https://github.com/StagPython/StagPy',

    author='Martina Ulvrova, Adrien Morison, Stéphane Labrosse',
    author_email='adrien.morison@gmail.com',

    license='Apache',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    python_requires='>=3.6',
    packages=['stagpy'],
    entry_points={
        'console_scripts': ['stagpy = stagpy.__main__:main']
    },
    include_package_data=True,
    install_requires=[
        'loam>=0.4.0',
        'f90nml>=1.3.1',
        'setuptools_scm>=6.2',
        'numpy>=1.19',
        'scipy>=1.5',
        'pandas>=1.1',
        'h5py>=3.1',
        'matplotlib>=3.3',
    ],
)
