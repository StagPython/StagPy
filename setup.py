from setuptools import setup

with open('README.rst') as rdm:
    LONG_DESCRIPTION = rdm.read()

setup(
    name='stagpy',
    version='0.1.0',

    description='Tool for StagYY output files processing',
    long_description=LONG_DESCRIPTION,

    url='https://github.com/mulvrova/StagPy',

    author='Martina Ulvrova, Adrien Morison, Stéphane Labrosse',

    license='GPLv2',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v2 (GPLv2)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        ],

    packages = ['stagpy'],
    entry_points = {
        'console_scripts': ['stagpy = stagpy.stagpy:main']
        },
    install_requires = [
        'numpy',
        'scipy',
        'f90nml',
        'matplotlib',
        'seaborn',
        'argcomplete',
        ],
)
