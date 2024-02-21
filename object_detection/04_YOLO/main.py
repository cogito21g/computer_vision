from ultralytics import YOLO
import cv2


# Load a model
model = YOLO("yolov8m-pose.pt")  


results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# Load Video File or Camera
capture = cv2.VideoCapture(0)
while cv2.waitKey(10) < 0:
    if capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT):
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    ret, frame = capture.read()
    cv2.imshow("Video Frame", frame)

capture.release()
cv2.destroyAllWindows()