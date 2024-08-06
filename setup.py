from setuptools import setup, find_packages

setup(
    name='dfx',      
    version='0.0.1',
    install_requires=[
        'numpy',
        'pandas',
    ],
    description='',
    author='Orazio Pontorno',
    author_email='orazio.pontorno@phd.unict.it',
    package_dir={'': 'src'},
    packages=find_packages(
        where='src'
    ),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)