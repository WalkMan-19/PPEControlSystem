from app.detection.singleton import init_detector
import cv2

detector = init_detector()

img = cv2.imread("test/images/image5.jpg")
results = detector.detect(img)
print(results)