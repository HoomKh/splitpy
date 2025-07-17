from setuptools import setup, find_packages
from pathlib import Path
from splitpy.version import __version__
from splitpy.authority import author, author_email

here = Path(__file__).resolve().parent.name

setup(
    name=here,
    version=__version__,
    author=author,
    author_email=author_email,
    packages=find_packages(),
    install_requires=["langchain","langchain-community", "pypdf", "tiktoken"],
    python_requires=">=3.8",
)