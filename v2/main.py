
import ultralytics
from ultralytics import YOLO
import cv2 as cv
import funcoes_desenho

print(ultralytics.checks())
model = YOLO('yolov8n.pt')


def show_pic(img):
    cv.imshow('image', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


img_path = '../v1/data/dog.jpg'
img = cv.imread(img_path, 1)

results = model.predict(img)

# bounding boxes parameters
for result in results:
    print(result.boxes.data)

# confidence
for result in results:
    print(result.boxes.conf)

# drawing boxes
funcoes_desenho.desenha_caixas(img, results[0].boxes.data)

# showing result
show_pic(img)
