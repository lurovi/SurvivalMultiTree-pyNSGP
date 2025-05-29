import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='pynsgp',
    version='1.5.0',
    author='Marco Virgolin, Luigi Rovito',
    author_email='luigirovito2@gmail.com',
    url='https://github.com/lurovi/SurvivalMultiTree-pyNSGP',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]

)
