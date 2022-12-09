# LabelMe2Yolov7Segmentation

This repository was designed in order to label images using [LabelMe](https://github.com/wkentaro/labelme) and transform to [YoloV7](https://github.com/WongKinYiu/yolov7) format for instance segmentation

## Instalation

`pip install labelme2yolov7segmentation`

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
