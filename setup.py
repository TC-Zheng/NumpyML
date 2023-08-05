from setuptools import setup, find_packages

setup(
    name='numpyml',
    version='0.1',
    description='Building the foundations of machine learning from scratch using only numpy.',
    author='Tianchen Zheng',
    author_email='eric.tc.zheng@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy==1.25.2',
    ],
)