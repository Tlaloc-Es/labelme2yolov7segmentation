import json
import pathlib
from setuptools import setup
from pip._internal.req import parse_requirements

with open("cz.json", "r", encoding="utf-8") as f:
    cz_file = json.load(f)

install_reqs = [*parse_requirements("./requirements.txt", session=False)]

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="labelme2yolov7segmentation",
    version=cz_file["commitizen"]["version"],
    description="Convert LabelMe format to yolov7 for segmentation.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/Tlaloc-Es/labelme2yolov7segmentation",
    author="Tlaloc-Es",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    packages=["labelme2yolo"],
    # packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[str(requirement.requirement) for requirement in install_reqs],
    entry_points={
        "console_scripts": [
            "labelme2yolo=labelme2yolo.__main__:main",
        ]
    },
)
