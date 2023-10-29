from ultralytics import YOLO
import cv2
import pandas as pd
import os



#model
qc_model = YOLO("best.pt") #qc_model = model.predict(source="0", show=True, conf = 0.2)

#open camera 
cap = cv2.VideoCapture(0)
 
vehicles = [0, 1]

while (True):
    ret, frame = cap.read()
#detection 
    detections = qc_model.track(frame, persist = True ,conf = 0.2)[0]
    annottated_frame = detections[0].plot()
    detections_ = []
    cv2.imshow("show", annottated_frame)
    if cv2.waitKey(1) & 0xFF == ord("f"):
        break
    for detection in detections.boxes.data.tolist() :
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles: 
            detections_.append([ x1, y1, x2, y2, score, class_id])
           # print(detections_)    
             

cap.release()
cv2.destroyAllWindows()   


   
   




    
