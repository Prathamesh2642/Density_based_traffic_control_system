import threading
from ultralytics import YOLO
import cv2    #use to capture and display images and perform manipulations on them
import cvzone # to display detections- also display fancy rectangle
import math

droidcam_ip1 = '192.168.137.60'
droidcam_port1 = '4747'
droidcam_ip2 = '192.168.137.8'
droidcam_port2 = '4747'
droidcam_ip3 = '192.168.137.248'
droidcam_port3 = '4747'
droidcam_ip4 = '192.168.137.32'
droidcam_port4 = '4747'


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


# Construct the DroidCam URL
droidcam_url1 = f'http://{droidcam_ip1}:{droidcam_port1}/video'
droidcam_url2 = f'http://{droidcam_ip2}:{droidcam_port2}/video'
droidcam_url3 = f'http://{droidcam_ip3}:{droidcam_port3}/video'
droidcam_url4 = f'http://{droidcam_ip4}:{droidcam_port4}/video'


class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Camera not present")
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        results1 = model(frame, stream=True)  # stream =True makes use of generators and hence is more efficient

        for i in results1:
            boundingboxes = i.boxes
            for j in boundingboxes:
                # open cv method or cv2 method
                '''
                x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                '''

                # cvzone method- more fancier bboxes
                x1, y1, x2, y2 = j.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(255, 0, 200))

                confidence = math.ceil((j.conf[0] * 100)) / 100
                print(confidence)

                category = int(j.cls[0])

                cvzone.putTextRect(frame, f'{classNames[category]} {confidence}', (max(0, x1), max(30, y1)), scale=2,
                                   thickness=1)


        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("Camera 1", droidcam_url1)
thread2 = camThread("Camera 2", droidcam_url2)
thread3 = camThread("Camera 3", droidcam_url3)
thread4 = camThread("Camera 4", droidcam_url4)


thread1.start()
thread2.start()
thread3.start()
thread4.start()