# This code is heavily inspired by: Murtaza Hassan
# (YouTube video: https://www.youtube.com/watch?v=HXDD7-EnGBY)
# Code changed and commented by: Rui Monteiro

import cv2


# Threshold to detect an object
thres = 0.5

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

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            # Insert box outlining object(s)
            cv2.rectangle(img, box, color=(0,255,0), thickness=2)
            
            # Insert the name of the class(es) and the confidence we have on the classification
            cv2.putText(img, classNames[classId-1].capitalize() + ' (' + str(round(confidence*100, 1)) + '%)',
            (box[0]+10, box[1]+30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
    cv2.imshow('Output', img)
    cv2.waitKey(1)
