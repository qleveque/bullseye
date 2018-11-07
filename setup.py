import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bullseye_method",
    version="0.0.2",
    author="Quentin Lévêque, Guillaume Dehaene",
    author_email="qleveque@hotmail.com, guillaume.dehaene@gmail.com",
    description=\
    "Implemented tensorflow version of the Bullseye method for prior approximation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Whenti/bullseye",
    packages=setuptools.find_packages(),
    install_requires = [
    "numpy>=1.15",
    "pandas>=0.23",
    "tensorflow>=1.10",
    "seaborn>=0.9.0",
    "matplotlib>=2.2.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)