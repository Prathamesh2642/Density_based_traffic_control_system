# import cv2
# from ultralytics import YOLO
# import cv2    #use to capture and display images and perform manipulations on them
# import cvzone # to display detections- also display fancy rectangle
# import math
# import torch
# # model=YOLO(r'D:\Learning only\Trial\yolotrial\best.pt')
# # model=YOLO(r'D:\Learning only\Trial\yolotrial\yolov5trained\best.pt')
# model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=r'D:\Learning only\Trial\yolotrial\yolov5trained\best.pt',force_reload=True).autoshape()
#
#
#
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"] #len = 80 categories
#
#
# droidcam_ip = '192.168.137.8'
# droidcam_port = '4747'
#
# # Construct the DroidCam URL
# droidcam_url = f'http://{droidcam_ip}:{droidcam_port}/video'
#
#
# # Open the video capture object
# cap = cv2.VideoCapture(droidcam_url)
#
# if not cap.isOpened():
#     print("Error: Could not open DroidCam.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     results = model(frame, stream=True)  # stream =True makes use of generators and hence is more efficient
#     if not ret:
#         print("Error: Failed to grab frame.")
#         break
#     for i in results:
#         boundingboxes=i.boxes
#         for j in boundingboxes:
#             # open cv method or cv2 method
#             '''
#             x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
#             x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
#             print(x1,y1,x2,y2)
#             cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
#             '''
#
#             #cvzone method- more fancier bboxes
#             x1,y1,x2,y2=j.xyxy[0]
#             x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
#             w,h=x2-x1,y2-y1
#
#             cvzone.cornerRect(frame,(x1,y1,w,h),colorC=(255,0,200))
#
#             confidence=math.ceil((j.conf[0]*100))/100
#             print(confidence)
#
#             category=int(j.cls[0])
#
#
#             cvzone.putTextRect(frame,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=2,thickness=1)
#
#     # Display the frame
#     cv2.imshow("DroidCam Feed", frame)
#
#     # Break the loop if 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as F
from matplotlib import pyplot as plt
import cv2

# Load YOLOv5 model
weights_path = r'D:\Learning only\Trial\yolotrial\yolov5trained\best.pt'  # Replace with the path to your custom-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=weights_path)

# Set video file path
video_path = r"D:\Learning only\yolo\Vehicles stuck in flood waters.mp4"  # Replace with the path to your custom video file
cap = cv2.VideoCapture(video_path)

# Process each frame in the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV BGR format to RGB
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Perform inference
    results = model(image)

    # Visualize the frame with bounding boxes
    img_with_boxes = results.render()[0]
    plt.imshow(img_with_boxes)
    plt.show()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
