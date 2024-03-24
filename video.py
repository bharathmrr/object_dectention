import cv2
config_file=r"C:\Users\gowth\pro\New folder\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
forxen=r"C:\Users\gowth\pro\New folder\frozen_inference_graph.pb"
model=cv2.dnn_DetectionModel(forxen,config_file)
model.setInputSize(320,320)

model.setInputScale(1.0/256.0)
model.setInputMean((125.5,127.5,127.5))
model.setInputSwapRB(True)
classlabels=[]
filename=r"C:\Users\gowth\pro\lables.txt"
with open(filename,'rt') as fpt:
    classlabels=fpt.read().rstrip('\n').split('\n')
model.setInputSize(320,320)
cap=cv2.VideoCapture(0)
#cap=cv2.VideoCapture(r"C:\Users\gowth\pro\New folder\WhatsApp Video 2024-03-24 at 13.42.48_ab88ba05.mp4")
if not cap.isOpened():
    cap=cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError('CANNOT open video ')
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
while True:
    ret,frame=cap.read()
    classIndex,confidence,bbox=model.detect(frame,confThreshold=0.55)
    print(classIndex)
    if (len(classIndex)!=0):
        for classIdx,conf,boxes in zip(classIndex.flatten(),confidence.flatten(),bbox):
            if(classIdx<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classlabels[classIdx-1],(boxes[0]+10,boxes[1]+40),font,3,color=(0,255,2),thickness=3)
    cv2.imshow("object detenction ",frame)
    if cv2.waitKey(2)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()