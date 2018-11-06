import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="bullseye_method",
    version="0.0.1",
    author="Quentin Lévêque, Guillaume Dehaene",
    author_email="qleveque@hotmail.com, guillaume.dehaene@gmail.com",
    description=\
    "Implemented tensorflow version of the Bullseye method for prior approximation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Whenti/bullseye",
    packages=setuptools.find_packages(),
    install_requires = [
    "numpy",
    "pandas",
    "tensorflow"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)