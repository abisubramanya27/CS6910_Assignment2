# PART C : YOLO v3 Object Detection using OPENCV
YOLO v3 is used for detecting objects and their locations in images and recorded or real-time videos. It is popular for it's accuracy and speed. (here we use OPENCV to use it)
The first paper on YOLO is shown <a href="https://arxiv.org/pdf/1506.02640.pdf">here</a> and improvements in v3 is decribed <a href="https://arxiv.org/pdf/1506.02640.pdf">here</a>

## Overview
The `src/main.py` is a python program that contains the necessary code to use the pre-trained yolov3 model for object detection on an input image, video or a live webcam feed and store the results.

There are several pre-trained models available and we would be using the "YOLOv3–416" model. The models are trained on the MS COCO dataset which has 80 classes of objects present in it.
- `pre-trained-model/coco.names` contains the 80 class names used in YOLO.
- `pre-trained-model/yolov3.cfg` is the configuration file for YOLOv3 (to use a different model, download the configuration file; other variants and versions of yolo can be downloaded from [link](https://pjreddie.com/darknet/yolo/))
- The weights file for the YOLOv3 model can be downloaded using the command `wget https://pjreddie.com/media/files/yolov3.weights` (those for other models can be downloaded similarly)


The code for this part is based on this [Medium](https://towardsdatascience.com/object-detection-using-yolov3-9112006d1c73) article.

## Requirements
The required python libraried can be installed using 
```shell
pip install -r requirementsC.txt
```

## Arguments
The `main.py` program is run with the following arguments
```shell
    --video          --->   Path to video file (default : None)
    --image          --->   Path to the test images (default : None)
    -camera          --->   To use the live feed from web-cam (default : False)
    --weights        --->   Path to model weights (default : ../pre-trained-model/yolov3.weights)
    --configs        --->   Path to model configs (default : ../pre-trained-model/yolov3.cfg)
    --class_names    --->   Path to class-names text file ../pre-trained-model/coco.name
    --conf_thresh    --->   Confidence threshold value (default : 0.6)
    --nms_thresh     --->   NMS (Non-maximum supression) threshold value (default : 0.4)
    -ds              --->   To display probability scores for the object detected in the output image/video (default : False)   
```
## Usage
* For testing on images  
`python _main.py --image <path to the image file> --weights <path to the weights file> --configs <path to the config file> --class_names <path to the class ids file>`

* For testing on videos  
`python main.py --video <path to the video file> --weights <path to the weights file> --configs <path to the config file> --class_names <path to the class ids file>`

* For testing on live web-cam feed  
`python Obj_main.py -camera --weights <path to the weights file> --configs <path to the config file> --class_names <path to the class ids file>`

The program will store the output image/video in the same name as input inside `output` folder, which will be created/existing one directory above the input image. (output for webcam feed will be stored in the name 'webcam_out.avi')

The output file formats :
- image : same format as input
- video/webcam :   `.avi`

## Output
Example output (with scores) for image (input image : `input/input1.png`)
<img src ='output/input1.png' width = 500>

Example output (without scores) for image (input image : `input/input2.jpg`)
<img src ='output/input2.jpg' width = 500>

Output for video (input video : `input/video1.mp4`) is uploaded to youtube
<a href="https://youtu.be/TpRf-LY3k4c" target="_blank"><img src="http://img.youtube.com/vi/TpRf-LY3k4c/0.jpg" 
alt="YOLOv3 output video" width="500"/></a>
