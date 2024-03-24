import cv2
import matplotlib.pyplot as plt

config_file=r"C:\Users\gowth\pro\New folder\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_file=r"C:\Users\gowth\pro\New folder\frozen_inference_graph.pb"
model=cv2.dnn_DetectionModel(frozen_file,config_file)
classlabels=[]
filename="lables.txt"
with open(filename,'rt') as fpt:
    classlabels=fpt.read().rstrip('\n').split('\n')
model.setInputSize(320,320)

model.setInputScale(1.0/256.0)
model.setInputMean((125.5,127.5,127.5))
model.setInputSwapRB(True)
img=cv2.imread(r"C:\Users\gowth\pro\New folder\Man-Pushing-a-Car-Credit-iStock-ArtmannWitte.jpg")
classIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for classInd,conf,boxes in zip(classIndex,confidence,bbox):
    cv2.rectangle(img,boxes,(255,0,0))
    cv2.putText(img,classlabels[classInd-1],(boxes[0]+40,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
cv2.imshow('image',img)
cv2.waitKey(5000)