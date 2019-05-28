
from io import open
from setuptools import find_packages, setup

setup(
    name="uda_bert",
    version="0.0.1",
    author="Bastien van Delft",
    author_email="bastien.van-delft@unchartech.com",
    description="PyTorch version of Unsupervised Data Augmentation using Bert as core model",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords='BERT NLP deep learning UDA data augmentation unsupervised',
    license='Apache',
    url="https://github.com/bhacquin/Unsupervised_Data_Augmentation",
    packages=find_packages(exclude=["*.tests", "*.tests.*",
                                    "tests.*", "tests"]),
    install_requires=['torch>=0.4.1',
                      'numpy',
                      'boto3',
                      'requests',
                      'tqdm',
                      'regex',
                      'googletrans',
                      'argparse'],
    entry_points={
      'console_scripts': [
        "uda_bert=uda_bert.__main__:main",
      ]
    },
    # python_requires='>=3.5.0',
    tests_require=['pytest'],
    classifiers=[
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
