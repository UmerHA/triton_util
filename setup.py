from setuptools import setup, find_packages

setup(
    name='triton-util',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['triton'],
    author='Umer Adil',
    author_email='umer.hayat.adil@gmail.com',
    description='Make Triton easier - A utility package for OpenAI Triton',
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    url='https://github.com/umerHA/triton_util',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
