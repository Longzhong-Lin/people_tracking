# People Tracking

## usage

1. run ROS
```
roscore
```

2. run Python script
```
conda activate yolov8

python people_tracking.py
```

## pipeline

1. 2D detection using YOLO
2. 2D tracking using DeepSORT
3. stereo depth estimation using UniMatch