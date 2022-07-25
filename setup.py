from setuptools import setup, find_packages

from pathlib import Path


def get_version() -> str:
    version_file = Path(__file__).parent / "rectools" / "version.py"
    content = version_file.read_text()
    exec(content)
    return locals()["VERSION"]


long_description = Path("README.md").read_text()
extras_path = Path("extras")

install_requires = Path("requirements.txt").read_text().split("\n")
nn_requires = (extras_path / "requirements-nn.txt").read_text().split("\n")


setup(
    name="rectools",
    version=get_version(),
    author="Potapov Daniil",
    author_email="mars-team@mts.ru",
    url="https://github.com/MobileTeleSystems/RecTools",
    description="An easy-to-use Python library for building recommendation systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["rectools", "rectools.*"]),
    install_requires=install_requires,
    extras_requires={
        "nn": [nn_requires],
    },
    python_requires=">=3.7",
)
