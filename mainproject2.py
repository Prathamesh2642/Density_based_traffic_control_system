#imports
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
import Sort_file
import time


#declarations

videoslist=[r'D:\Learning only\Random Python projects\tr1.mp4',r'D:\Learning only\Random Python projects\tr2.mp4',
            r'D:\Learning only\Random Python projects\tr3.mp4']

masklist=[r'D:\Learning only\Random Python projects\ms1.png',r'D:\Learning only\Random Python projects\ms2.png',
            r'D:\Learning only\Random Python projects\ms3.png',]

model=YOLO('../weights/yolov8l.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

detectionlist=['car','bus','motorbike','truck']

coords={'v1':[0,330,300,330],'v2':[0,330,600,330],'v3':[0,330,600,330]}

tracker=Sort_file.Sort(max_age=20,min_hits=3,iou_threshold=0.3)

density_tracker_dict={'track_1':{'car':0,'bike':0,'truck':0,'bus':0},
                      'track_2':{'car':0,'bike':0,'truck':0,'bus':0},
                      'track_3':{'car':0,'bike':0,'truck':0,'bus':0}
                      }

# cap.set(3,1280)
# cap.set(4,720)

# for loop starts here - for number of videos
for i in range(len(videoslist)):
    t0 = time.time()  # start time in seconds
    num_seconds = 0

    cap=cv2.VideoCapture(videoslist[i])
    mask=cv2.imread(masklist[i])


# classNames=['bus','car','motorbike','truck']


# limits4=[280,300,1200,300]
    totalcount=[]
    cardetection=[]
    busdetection=[]
    bikedetection =[]
    truckdetection =[]
    while True:
        success,img=cap.read()
        imgregion=cv2.bitwise_and(img,mask)
        results=model(imgregion,stream=True)
        detections=np.empty((0,6))


        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

                w,h=x2-x1,y2-y1
    #         show confidence leve
                conf=math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                currentclass=classNames[cls]

                if ((currentclass in detectionlist) and (conf > 0.30)):

                    cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=2)

                    cvzone.putTextRect(img, f'{currentclass} {conf}', (max(0, x1), max(35, y1)),scale=0.9,thickness=1,offset=2)
                    currenct_array=np.array([x1,y1,x2,y2,conf,detectionlist.index(currentclass)])
                    detections=np.vstack((currenct_array,detections))
                # print(detections)

        # cv2.line(img,(limits1[0],limits1[1]),(limits1[2],limits1[3]),(0,0,255),5)
        cv2.line(img,(coords[f'v{i+1}'][0],coords[f'v{i+1}'][1]),(coords[f'v{i+1}'][2],coords[f'v{i+1}'][3]),(0,0,255),5)

    # print(detections)
    #     coords[f'v{i}'][0]
        result_tracker=tracker.update(detections)
    # print("sadfksdljsl;fjasjfoksdjokfnvoskniowf35928572385327894723849752398075230958275832")
    # print(result_tracker)

        for result in result_tracker:
            # print(result)
            x1,y1,x2,y2,currentclass,id=result
            # y2=li[0]
            # currentclass=li[1]

            x1,y1,x2,y2,currentclass=int(x1),int(y1),int(x2),int(y2),int(currentclass)
            w,h=x2-x1,y2-y1
            # print(result)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, t=2,colorR=(255,0,0))
            cvzone.putTextRect(img, f'{id} ', (max(0, x1), max(35, y1)), scale=0.9, thickness=1, offset=2)
    #     circle at each object
            cx,cy=x1+w//2,y1+h//2
            cv2.circle(img,(cx,cy),3,(55,200,20),cv2.FILLED)
            if(coords[f'v{i+1}'][0]<cx<coords[f'v{i+1}'][2]) and (coords[f'v{i+1}'][1]-30<cy<coords[f'v{i+1}'][3]+30):

                if (currentclass == 0):
                    if (cardetection.count(id) == 0 and busdetection.count(id)==0 and truckdetection.count(id)==0 and bikedetection.count(id)==0):
                        cardetection.append(id)

                elif (currentclass == 1):
                    if (cardetection.count(id) == 0 and busdetection.count(id)==0 and truckdetection.count(id)==0 and bikedetection.count(id)==0):
                        busdetection.append(id)

                elif (currentclass == 3):
                    if (cardetection.count(id) == 0 and busdetection.count(id)==0 and truckdetection.count(id)==0 and bikedetection.count(id)==0):
                        truckdetection.append(id)

                elif (currentclass == 2):
                    if (cardetection.count(id) == 0 and busdetection.count(id)==0 and truckdetection.count(id)==0 and bikedetection.count(id)==0):
                        bikedetection.append(id)
                if(totalcount.count(id)==0):
                    totalcount.append(id)
            cvzone.putTextRect(img, f'Vehicle Count : {len(totalcount)}  Cars :{len(cardetection)} Bus: {len(busdetection)} Truck :{len(truckdetection)} Bike:{len(bikedetection)}',(0,50) ,scale=1.5, thickness=2, offset=3)



        t1 = time.time()  # current time
        num_seconds = t1 - t0  # diff
        if num_seconds >= 12:  # e.g. break after 12 seconds
            break
        cv2.imshow('image',img)
        # cv2.imshow('imageregion',imgregion)
        cv2.waitKey(1)
    density_tracker_dict[f'track_{i + 1}']['car'] = density_tracker_dict[f'track_{i + 1}']['car'] + len(cardetection)
    density_tracker_dict[f'track_{i + 1}']['bus'] = density_tracker_dict[f'track_{i + 1}']['bus'] + len(busdetection)
    density_tracker_dict[f'track_{i + 1}']['truck'] = density_tracker_dict[f'track_{i + 1}']['truck'] + len(truckdetection)
    density_tracker_dict[f'track_{i + 1}']['bike'] = density_tracker_dict[f'track_{i + 1}']['bike'] + len(bikedetection)
print(density_tracker_dict)
