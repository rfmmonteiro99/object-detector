# This version applies Non-maximum Suppression (NMS) to filter the predictions of this object detector.
# To know more about NMS check: https://paperswithcode.com/method/non-maximum-suppression

import cv2
import numpy as np


# Threshold to detect an object
thres = 0.5

# NMS threshold
nms_threshold = 0.5

# Capture video (note: this sets will depend from computer to computer)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)

# Get all the classes in a list: ['person', 'bicycle', 'car', ..., 'hair brush']
classNames = []
classFile = 'data/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Get paths to the pre-trained network
configPath = 'data/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'data/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)

net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Use the Object Detector on the video provided by the camera
while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print('Object ID:', classIds, '; Box:', bbox)

    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        # Insert box outlining object(s)
        cv2.rectangle(img, (x,y), (x+w,h+y), color=(0, 255, 0), thickness=2)
        
        # Insert the name of the class(es)
        cv2.putText(img, classNames[classIds[i]-1].capitalize(), 
        (box[0]+10, box[1]+30),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('Output', img)
    cv2.waitKey(1)
