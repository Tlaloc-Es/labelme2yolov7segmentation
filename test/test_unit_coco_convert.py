import os
import pathlib
import json
import numpy as np
from labelme2yolo.__main__ import process


def test_conversion():

    current_path = pathlib.Path(__file__).parent.resolve()

    with open(
        os.path.join(current_path, "__data__/yolov7_annotations/000000000285.txt"),
        encoding="utf-8",
    ) as file_handler:
        data = file_handler.readline()
        data = data.replace("\n", "")
        numbers = data.split(" ")[1:]
        numbers = np.array([*map(float, numbers)])

    with open(
        os.path.join(current_path, "__data__/yolov7_annotations/000000000285.json"),
        encoding="utf-8",
    ) as file_handler:
        data = json.load(file_handler)
        height = int(data["imageHeight"])
        width = int(data["imageWidth"])
        values = np.array(data["values"])

    converted_values = process(values, width, height)

    for x0, x1 in zip(numbers, converted_values):
        assert x0 - x1 < 0.000001
