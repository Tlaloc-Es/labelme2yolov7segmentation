from math import log10, floor
import itertools
import os
import json
import shutil
from pathlib import Path
from glob import glob, iglob
from typing import List, Optional
import numpy as np
from numpy import ndarray
import click
from datatypes import (
    LabelMe,
    OutputPaths,
    Polygon,
    ShapeProcessed,
    ShapesProcessed,
    FileNameAndExtension,
)

ALLOWED_SHAPES = ["polygon"]


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def round_sig(number: float, sig: int = 6):
    return round(number, sig - int(floor(log10(abs(number)))) - 1)


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
    with open(
        os.path.join(output_path, "labels.txt"), "w", encoding="utf-8"
    ) as file_writer_handler:
        for i, label in enumerate(labels):
            file_writer_handler.write(f"{label}:{i}\n")


def write_shapes(
    source_path: str,
    shapes_processed: ShapesProcessed,
    output_paths: OutputPaths,
    train_percentage: int = 70,
) -> None:
    shapes = np.array(shapes_processed.shapes)
    index_to_split = int((len(shapes) * train_percentage) / 100)

    shapes = np.array(shapes_processed.shapes)
    np.random.shuffle(shapes)
    train, val = shapes[:index_to_split], shapes[index_to_split:]

    shapes_processed_train = ShapesProcessed(shapes=train.tolist())
    shapes_processed_val = ShapesProcessed(shapes=val.tolist())

    process_write_shapes(
        source_path,
        shapes_processed_train,
        output_paths.images_train_path,
        output_paths.labels_train_path,
        output_paths.dataset_path,
        "train.txt",
    )
    process_write_shapes(
        source_path,
        shapes_processed_val,
        output_paths.images_val_path,
        output_paths.labels_val_path,
        output_paths.dataset_path,
        "test.txt",
    )


def get_file_name_and_extension(path: str) -> FileNameAndExtension:
    split_source_image_path = os.path.splitext(path)
    file_name = Path(path).stem
    extension = split_source_image_path[1]
    return FileNameAndExtension(file_name=file_name, extension=extension)


def process_write_shapes(
    source_path: str,
    shapes_processed: ShapesProcessed,
    image_output_path: str,
    points_output_path: str,
    dataset_path: str,
    label_file_name: str,
):

    for shape in shapes_processed.shapes:
        source_image_path = get_image_path(source_path, shape.file_name)
        file_name_and_extension = get_file_name_and_extension(source_image_path)

        output_image_path = os.path.join(
            image_output_path,
            f"{file_name_and_extension.file_name}{file_name_and_extension.extension}",
        )

        shutil.copyfile(source_image_path, output_image_path)

        for pyligon in shape.polygons:
            with open(
                os.path.join(points_output_path, f"{shape.file_name}.txt"),
                "a",
                encoding="utf-8",
            ) as file_append_handler:
                file_append_handler.write(f"{pyligon.get_representation()}\n")

            with open(
                os.path.join(dataset_path, label_file_name),
                "a",
                encoding="utf-8",
            ) as file_append_handler:
                file_append_handler.write(f"{output_image_path}\n")


def get_image_path(source_path: str, file_name: str) -> Optional[str]:
    files = glob(os.path.join(source_path, f"{file_name}.jpg"))
    files.extend(glob(os.path.join(source_path, f"{file_name}.jpeg")))
    files.extend(glob(os.path.join(source_path, f"{file_name}.png")))

    if len(files) == 0:
        return None

    return files[0]


def read_labelme_file(path: str) -> LabelMe:
    with open(path, "r", encoding="utf-8") as file_reader_handler:
        data = json.load(file_reader_handler)
        label_me_value = LabelMe(
            **data, image_height=data["imageHeight"], image_width=data["imageWidth"]
        )
    return label_me_value


def create_dataset(output_path: str) -> OutputPaths:
    images_path = os.path.join(output_path, "images")
    images_train_path = os.path.join(images_path, "train")
    images_val_path = os.path.join(images_path, "val")

    labels_path = os.path.join(output_path, "labels")
    labels_train_path = os.path.join(labels_path, "train")
    labels_val_path = os.path.join(labels_path, "val")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_val_path, exist_ok=True)

    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(labels_train_path, exist_ok=True)
    os.makedirs(labels_val_path, exist_ok=True)

    return OutputPaths(
        dataset_path=output_path,
        images_path=images_path,
        images_train_path=images_train_path,
        images_val_path=images_val_path,
        labels_path=labels_path,
        labels_train_path=labels_train_path,
        labels_val_path=labels_val_path,
    )


@click.command()
@click.option(
    "--source-path", required=True, help="Path of the labelme project with json"
)
@click.option("--output-path", required=True, help="Path of the yolov7 format files")
def main(source_path: str, output_path: str):
    labels = []
    shapes_processed = ShapesProcessed()

    for path in iglob(os.path.join(source_path, "*.json")):
        file_name = Path(path).stem
        label_me_value = read_labelme_file(path)
        polygons: List[Polygon] = []

        for shape in label_me_value.shapes:
            if shape.shape_type in ALLOWED_SHAPES:
                if shape.label not in labels:
                    labels.append(shape.label)

                label_index = labels.index(shape.label)

                points = process(
                    np.array(flatten(shape.points)),
                    label_me_value.image_width,
                    label_me_value.image_height,
                )

                polygon = Polygon(
                    label_index=label_index,
                    label_name=shape.label,
                    points=points,
                )
                polygons.append(polygon)

        shape_processed = ShapeProcessed(
            path=path, file_name=file_name, polygons=polygons
        )

        shapes_processed.shapes.append(shape_processed)

    output_paths = create_dataset(output_path)

    write_shapes(source_path, shapes_processed, output_paths)

    write_labels(labels, output_path)


if __name__ == "__main__":
    main()
