#
from setuptools import find_packages, setup

setup(
    #
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"whisper_finetune": ["py.typed"]},
)
