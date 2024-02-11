import threading
import cv2

droidcam_ip1 = '192.168.137.60'
droidcam_port1 = '4747'
droidcam_ip2 = '192.168.137.8'
droidcam_port2 = '4747'


# Construct the DroidCam URL
droidcam_url1 = f'http://{droidcam_ip1}:{droidcam_port1}/video'
droidcam_url2 = f'http://{droidcam_ip2}:{droidcam_port2}/video'

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
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

# Create two threads as follows
thread1 = camThread("Camera 1", droidcam_url1)
thread2 = camThread("Camera 2", droidcam_url2)
thread1.start()
thread2.start()