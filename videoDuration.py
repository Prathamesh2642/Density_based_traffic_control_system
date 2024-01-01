import cv2 as cv

cap = cv.VideoCapture(r"D:\\Learning only\\Random Python projects\\videoplayback.mp4")
fps = cap.get(cv.CAP_PROP_FPS)      # OpenCV v2.x used "CV_CAP_PROP_FPS"
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
duration = frame_count/fps

print('fps = ' + str(fps))
print('number of frames = ' + str(frame_count))
print('duration (S) = ' + str(duration))
minutes = int(duration/60)
seconds = duration%60
print('duration (M:S) = ' + str(minutes) + ':' + str(seconds))

cap.release()
