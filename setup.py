from setuptools import find_packages, setup

setup(
    name="tf-bert",
    version="1.0.0a0",
    install_requires=["tensorflow>=2"],
    packages=find_packages(exclude=["tests"]),
    python_requires=">=3.6, <3.8",
    #
    description="bert implementation",
    author="Jeong Ukjae",
    author_email="jeongukjae@gmail.com",
    url="https://github.com/jeongukjae/tf-bert",
    classifiers=[
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)
