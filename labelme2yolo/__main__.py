from math import log10, floor
import itertools
import os
import json
from pathlib import Path
from glob import iglob
from typing import List
import numpy as np
from numpy import ndarray
from pydantic import BaseModel, conlist
import click


ALLOWED_SHAPES = ["polygon"]


class LabelMeShape(BaseModel):
    label: str
    points: List[conlist(float, min_items=2, max_items=2)]
    shape_type: str


class LabelMe(BaseModel):
    shapes: List[LabelMeShape]
    image_height: int
    image_width: int


def flatten(l):
    return [item for sublist in l for item in sublist]


def round_sig(x: float, sig: int = 6):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


def process(values: ndarray, width: float, height: float) -> List[float]:
    odd = values[0::2]
    even = values[1::2]

    odd_processed = odd / width
    even_processed = even / height

    return [
        *map(
            round_sig,
            list(itertools.chain(*[*zip(odd_processed, even_processed)])),
        )
    ]


def write_labels(labels: List[str], output_path: str) -> None:
    with open(os.path.join(output_path, "labels.txt"), "w", encoding="utf-8") as fw:
        for i, label in enumerate(labels):
            fw.write(f"{label}:{i}")


def write_shapes(
    output_path: str, file_name: str, shapes_processed: List[List[float]]
) -> None:
    with open(
        os.path.join(output_path, f"{file_name}.txt"), "w", encoding="utf-8"
    ) as fw:
        for shape in shapes_processed:
            fw.write(" ".join(map(str, shape)))


def read_labelme_file(path: str) -> LabelMe:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        label_me_value = LabelMe(
            **data, image_height=data["imageHeight"], image_width=data["imageWidth"]
        )
    return label_me_value


@click.command()
@click.option(
    "--source-path", required=True, help="Path of the labelme project with json"
)
@click.option("--output-path", required=True, help="Path of the yolov7 format files")
def main(source_path: str, output_path: str):
    labels = []
    for path in iglob(os.path.join(source_path, "*.json")):
        file_name = Path(path).stem
        label_me_value = read_labelme_file(path)

        shapes_processed = []

        for shape in label_me_value.shapes:
            if shape.shape_type in ALLOWED_SHAPES:
                if shape.label not in labels:
                    labels.append(shape.label)

                label_class = labels.index(shape.label)

                processed_values = process(
                    np.array(flatten(shape.points)),
                    label_me_value.image_width,
                    label_me_value.image_height,
                )

                processed_values.insert(0, label_class)
                shapes_processed.append(processed_values)

        write_shapes(output_path, file_name, shapes_processed)

    write_labels(labels, output_path)


if __name__ == "__main__":
    main()
