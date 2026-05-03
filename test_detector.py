from app.detection.singleton import init_detector
import cv2

detector = init_detector()
img = cv2.imread("image3.jpg")
results = detector.detect(img)
print(results)