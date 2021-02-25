from setuptools import setup

setup(
    name = 'bernina_analysis'
    version = '0.0.1'
    description = 'Analysis scripts for SwissFEL data'
    url = 'https://github.com/mankowsk/bernina_analysis'
    author='Roman Mankowsky',
    author_email='roman.mankowsky@psi.ch',
    packages=setuptools.find_packages(),
    requires=["numpy"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
         ),
)
