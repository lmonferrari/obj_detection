import os
import time
import cv2
import numpy as np


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


print('Opencv version: ', cv2.__version__)

config_path = './cfg'
coco_names = 'coco.names'
labels_path = os.path.join(config_path, coco_names)

with open(labels_path, 'r') as f:
    labels = f.read().strip().split('\n')

net = cv2.dnn.readNetFromDarknet(
    './cfg/yolov4.cfg', './heights/yolov4.weights')

# cores do bounding box
COLORS = np.random.randint(0, 255, (len(labels), 3), 'uint8')

# camadas
ln = net.getLayerNames()
# camadas de saida
output_layers = np.take(ln, net.getUnconnectedOutLayers() - 1)

# imagem de entrada
img = cv2.imread('./data/eagle.jpg', 1)
img_copy = img.copy()

H, W = img.shape[:2]
print(f'Height: {H} - Width {W}')

start = time.time()

blob = cv2.dnn.blobFromImage(
    img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

net.setInput(blob)
outputs = net.forward(output_layers)

end = time.time()
print(f'Elapsed time: {end - start}')

threshold = 0.5  # gatilho de confiança
threshold_NMS = 0.3  # non maximum supression
bounding_box = []
confidences = []
classes_id = []


for output in outputs:
    for detection in output:
        # os valores a partir da posição 5 são as porcentagens do objeto
        # detectado
        scores = detection[5:]
        # pegamos o index da maior porcentagem detectada
        class_id = np.argmax(scores)
        # pegamos a porgentagem
        confidence = scores[class_id]
        if confidence >= threshold:
            print(f'Class: {labels[class_id]} | '
                  f'Confidence: {confidence * 100 :.2f}%')

            (centerX, centerY, width,
             height) = detection[:4] * np.array([W, H, W, H])

            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))

            bounding_box.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            classes_id.append(class_id)

# non max supression
objs = cv2.dnn.NMSBoxes(bounding_box, confidences, threshold, threshold_NMS)

if len(objs):
    for obj in objs:
        x, y = bounding_box[obj][0], bounding_box[obj][1]
        w, h = bounding_box[obj][2], bounding_box[obj][3]
        # img_new = img_copy[y: y+h, x: x+w]
        colors = [int(c) for c in COLORS[classes_id[obj]]]

        cv2.rectangle(img, (x, y), (x + w, y + h), colors, 2)
        text = f'{labels[classes_id[obj]]} - '\
               f'{confidences[obj] * 100 :.2f}'
        cv2.putText(img, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors, 2)

show_image(img)
