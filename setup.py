# setup.py
from setuptools import setup
from setuptools import find_packages

# The version is updated automatically with bumpversion
# Do not update manually
__version = '0.1.0'

# windows installer:
# python setup.py bdist_wininst

# patch distutils if it can't cope with the "classifiers" or
# "download_url" keywords

# ipython profile magic commands implementation

setup(
    name='bl22cleardata',
    description='BL22 CLEAR post-processing',
    version=__version,
    author='Roberto J. Homs Puron',
    author_email='rhoms@cells.es',
    url='',
    packages=find_packages(),
    include_package_data=True,
    license="Python",
    long_description='Python library for the CLEAR Spectrometer '
                     'post-processing',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: GPLv3',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Topic :: Scientific Software',
        'Topic :: Software Development :: Libraries',
    ],
    entry_points={
        'console_scripts': [
            'pyClear = bl22cleardata.__main__:main',
        ]
    }
)
