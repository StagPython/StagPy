from setuptools import setup

setup(
    name = 'StagPy',
    packages = ['stagpy'],
    entry_points = {
        'console_scripts': ['stagpy = stagpy.stagpy:main']
        },
    version = '0.1',
    description = 'Tool for StagYY output files processing',
    url = 'https://github.com/mulvrova/StagPy',
    install_requires = [
        'numpy',
        'scipy',
        'f90nml',
        'matplotlib',
        'seaborn',
        'argcomplete',
        ],
    )
