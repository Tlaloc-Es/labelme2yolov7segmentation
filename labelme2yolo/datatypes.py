from typing import List
from pydantic import BaseModel, conlist


class LabelMeShape(BaseModel):
    label: str
    points: List[conlist(float, min_items=2, max_items=2)]
    shape_type: str


class LabelMe(BaseModel):
    shapes: List[LabelMeShape]
    image_height: int
    image_width: int


class OutputPaths(BaseModel):
    dataset_path: str
    images_path: str
    images_train_path: str
    images_val_path: str
    labels_path: str
    labels_train_path: str
    labels_val_path: str


class SplitedDataset(BaseModel):
    test: List[str]
    train: List[str]


class FileNameAndExtension(BaseModel):
    file_name: str
    extension: str


class Polygon(BaseModel):
    points: List[float]
    label_index: int
    label_name: str

    def get_representation(self):
        return str(self.label_index) + " " + " ".join(map(str, self.points))


class ShapeProcessed(BaseModel):
    path: str
    file_name: str
    polygons: List[Polygon]


class ShapesProcessed(BaseModel):
    shapes: List[ShapeProcessed] = []
