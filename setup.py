import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='pynsgp',
    version='1.3.0',
    author='Marco Virgolin',
    author_email='marco.virgolin@gmail.com',
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
