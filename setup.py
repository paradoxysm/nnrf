import os
from setuptools import setup, find_packages

def read(*paths):
    """
	Build a file path from *paths* and return the contents.
	"""
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

setup(
	name='nnrf',
	version='1.0.0',
    description='Neural Network with Random Forest Structure',
    long_description=(read('README.md') + '\n\n'),
	long_description_content_type="text/markdown",
	url='http://github.com/paradoxysm/nnrf',
	download_url = 'https://github.com/paradoxysm/nnrf/archive/1.0.0.tar.gz',
    author='paradoxysm',
	author_email='paradoxysm.dev@gmail.com',
    license='BSD-3-Clause',
    packages=find_packages(),
    install_requires=[
		'numpy',
		'tqdm'
    ],
	extras_require={
        'test': ['pytest', 'coverage', 'pytest-cov']
    },
	python_requires='>=3.4, <4',
	classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
	'Topic :: Scientific/Engineering :: Information Analysis',
	'Intended Audience :: Science/Research'
    ],
	keywords=['python','ml','neural-network','random-forest'],
    zip_safe=True)
