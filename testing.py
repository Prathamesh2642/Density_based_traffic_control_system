import cv2

# Replace 'your_droidcam_ip' and 'your_droidcam_port' with the actual IP address and port of your DroidCam.
droidcam_ip = 'your_droidcam_ip'
droidcam_port = 'your_droidcam_port'

# Construct the DroidCam URL
droidcam_url = f'http://192.168.137.199:4747/video'

# Open the video capture object
cap = cv2.VideoCapture(droidcam_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open DroidCam.")
    exit()

# Read and display frames from DroidCam
while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow("DroidCam Feed", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
