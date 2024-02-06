import cv2
from ultralytics import YOLO
import cv2    #use to capture and display images and perform manipulations on them
import cvzone # to display detections- also display fancy rectangle
import math

# cap=cv2.VideoCapture(0) #we can pass video location


model=YOLO(r'../yolo_weights/yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories


droidcam_ip1 = '192.168.0.103'
droidcam_port1 = '4747'
droidcam_ip2 = '192.168.0.102'
droidcam_port2 = '4747'


# Construct the DroidCam URL
droidcam_url1 = f'http://{droidcam_ip1}:{droidcam_port1}/video'
droidcam_url2 = f'http://{droidcam_ip2}:{droidcam_port2}/video'


# Open the video capture object
cap1 = cv2.VideoCapture(droidcam_url1)
cap1.set(3,640) #width  640 1280
cap1.set(4,480) #height 480 720

cap2 = cv2.VideoCapture(droidcam_url2)
cap2.set(3,640) #width  640 1280
cap2.set(4,480) #height 480 720



if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open DroidCam.")
    exit()

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    results1 = model(frame1, stream=True)  # stream =True makes use of generators and hence is more efficient
    results2=model(frame2,stream=True)

    if (not ret1) or (not ret2):
        print("Error: Failed to grab frame.")
        break
    for i in results1:
        boundingboxes=i.boxes
        for j in boundingboxes:
            # open cv method or cv2 method
            '''
            x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
            '''

            #cvzone method- more fancier bboxes
            x1,y1,x2,y2=j.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1

            cvzone.cornerRect(frame1,(x1,y1,w,h),colorC=(255,0,200))

            confidence=math.ceil((j.conf[0]*100))/100
            print(confidence)

            category=int(j.cls[0])


            cvzone.putTextRect(frame1,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=2,thickness=1)

    # Display the frame
    cv2.imshow("DroidCam Feed", frame1)
    cv2.imshow("DroidCam Feed", frame2)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap1.release()
cap2.release()
cv2.destroyAllWindows()
