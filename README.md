# LabelMe2Yolov7Segmentation

<div align="center">

[![Downloads](https://static.pepy.tech/personalized-badge/labelme2yolov7segmentation?period=month&units=international_system&left_color=grey&right_color=blue&left_text=PyPi%20Downloads)](https://pepy.tech/project/labelme2yolov7segmentation)
[![Stars](https://img.shields.io/github/stars/Tlaloc-Es/labelme2yolov7segmentation?color=yellow&style=flat)](https://github.com/Tlaloc-Es/labelme2yolov7segmentation/stargazers)

</div>

Convert [LabelMe](https://github.com/wkentaro/labelme) format into [YoloV7](https://github.com/WongKinYiu/yolov7) format for instance segmentation.

## Instalation [![PyPI](https://img.shields.io/pypi/v/labelme2yolov7segmentation.svg)](https://pypi.org/project/labelme2yolov7segmentation/)

You can install `labelme2yolov7segmentation` from [Pypi](https://pypi.org/project/labelme2yolov7segmentation/). It's going to install the library itself and its prerequisites as well.

```bash
pip install labelme2yolov7segmentation
```

You can install `labelme2yolov7segmentation` from its source code.

```bash
git clone https://github.com/Tlaloc-Es/labelme2yolov7segmentation.git
cd labelme2yolov7segmentation
pip install -e .
```

## Usage

First of all, make your dataset with LabelMe, after that call to the following command

`labelme2yolo --source-path /labelme/dataset --output-path /another/path`

The arguments are:

* `--source-path`: That indicates the path where are the json output of LabelMe and their images, both will have been in the same folder
* `--output-path`: The path where you will save the converted files and a copy of the images following the yolov7 folder estructure

### Expected output

If you execute the following command:

`labelme2yolo --source-path /labelme/dataset --output-path /another/datasets`

You will get something like this

```bash
datasets
├── images
│   ├── train
│   │   ├── img_1.jpg
│   │   ├── img_2.jpg
│   │   ├── img_3.jpg
│   │   ├── img_4.jpg
│   │   └── img_5.jpg
│   └── val
│       ├── img_6.jpg
│       └── img_7.jpg
├── labels
│   ├── train
│   │   ├── img_1.txt
│   │   ├── img_2.txt
│   │   ├── img_3.txt
│   │   ├── img_4.txt
│   │   └── img_5.txt
│   └── val
│       ├── img_6.txt
│       └── img_7.txt
├── labels.txt
├── test.txt
└── train.txt
```

## Donation

If you want to contribute you can make a donation at https://www.buymeacoffee.com/tlaloc, thanks in advance
