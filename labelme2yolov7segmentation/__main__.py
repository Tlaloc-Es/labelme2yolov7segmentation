import itertools
import json
import os
import shutil
from glob import glob, iglob
from math import floor, log10
from pathlib import Path
from typing import List, Optional

import click
import numpy as np
import yaml
from labelme2yolov7segmentation.datatypes import (
    FileNameAndExtension,
    LabelMe,
    OutputPaths,
    Polygon,
    ShapeProcessed,
    ShapesProcessed,
    YoloV7YML,
)
from numpy import ndarray

ALLOWED_SHAPES = ["polygon"]
TRAIN_TXT = "train.txt"
VAL_TXT = "val.txt"
TEST_TXT = "test.txt"


def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]


def round_sig(number: float, sig: int = 6):
    if number == 0:
        return 0.0
    return round(number, sig - int(floor(log10(abs(number)))) - 1)


def process(values: ndarray, width: int, height: int) -> List[float]:
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


def write_yolov7_yml(yolov7_yml: YoloV7YML, output_path: str) -> None:
    with open(os.path.join(output_path, "project.yml"), "w", encoding="utf-8") as file:
        yaml.dump(yolov7_yml.dict(), file)


def write_shapes(
    source_path: str,
    shapes_processed: ShapesProcessed,
    output_paths: OutputPaths,
    train_percentage: int = 70,
    val_percentage: int = 20,
) -> None:
    shapes = np.array(shapes_processed.shapes)
    index_to_split_train = int((len(shapes) * train_percentage) / 100)
    index_to_split_test = (
        int((len(shapes) * val_percentage) / 100) + index_to_split_train
    )

    shapes = np.array(shapes_processed.shapes)
    np.random.shuffle(shapes)
    train, val, test = (
        shapes[:index_to_split_train],
        shapes[index_to_split_train:index_to_split_test],
        shapes[index_to_split_test:],
    )

    shapes_processed_train = ShapesProcessed(shapes=train.tolist())
    shapes_processed_val = ShapesProcessed(shapes=val.tolist())
    shapes_processed_test = ShapesProcessed(shapes=test.tolist())

    process_write_shapes(
        source_path,
        shapes_processed_train,
        output_paths.images_train_path,
        output_paths.labels_train_path,
        output_paths.dataset_path,
        TRAIN_TXT,
    )
    process_write_shapes(
        source_path,
        shapes_processed_val,
        output_paths.images_val_path,
        output_paths.labels_val_path,
        output_paths.dataset_path,
        VAL_TXT,
    )
    process_write_shapes(
        source_path,
        shapes_processed_test,
        output_paths.images_test_path,
        output_paths.labels_test_path,
        output_paths.dataset_path,
        TEST_TXT,
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

        output_image_path = output_image_path.replace(dataset_path, ".")

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
    with open(path, encoding="utf-8") as file_reader_handler:
        data = json.load(file_reader_handler)
        label_me_value = LabelMe(
            **data, image_height=data["imageHeight"], image_width=data["imageWidth"]
        )
    return label_me_value


def create_dataset(output_path: str) -> OutputPaths:
    images_path = os.path.join(output_path, "images")
    images_train_path = os.path.join(images_path, "train")
    images_val_path = os.path.join(images_path, "val")
    images_test_path = os.path.join(images_path, "test")

    labels_path = os.path.join(output_path, "labels")
    labels_train_path = os.path.join(labels_path, "train")
    labels_val_path = os.path.join(labels_path, "val")
    labels_test_path = os.path.join(labels_path, "test")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(images_train_path, exist_ok=True)
    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(images_test_path, exist_ok=True)

    os.makedirs(labels_path, exist_ok=True)
    os.makedirs(labels_train_path, exist_ok=True)
    os.makedirs(labels_val_path, exist_ok=True)
    os.makedirs(labels_test_path, exist_ok=True)

    return OutputPaths(
        dataset_path=output_path,
        images_path=images_path,
        images_train_path=images_train_path,
        images_val_path=images_val_path,
        images_test_path=images_test_path,
        labels_path=labels_path,
        labels_train_path=labels_train_path,
        labels_val_path=labels_val_path,
        labels_test_path=labels_test_path,
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

    yolov7_yml = YoloV7YML(
        train=os.path.join(output_paths.dataset_path, TRAIN_TXT),
        val=os.path.join(output_paths.dataset_path, VAL_TXT),
        test=os.path.join(output_paths.dataset_path, TEST_TXT),
        nc=len(labels),
        names=labels,
    )

    write_yolov7_yml(yolov7_yml, output_path)


if __name__ == "__main__":
    main()
