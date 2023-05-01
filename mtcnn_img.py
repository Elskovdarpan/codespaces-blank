import cv2
from mtcnn_cv2 import MTCNN

detector = MTCNN()
conf_t = 0.99
test_pic = "pic4.jpg"

image = cv2.cvtColor(cv2.imread(test_pic), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)

# Result is an array with all the bounding boxes detected. Show the first.
print(result)

# print(results)
for res in result:
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height

    confidence = res['confidence']
    if confidence < conf_t:
        continue
    key_points = res['keypoints'].values()

    cv2.rectangle(test_pic, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
    cv2.putText(test_pic, f'conf: {confidence:.3f}', (x1, y1), cv2.FONT_ITALIC, 1, (0, 0, 255), 1)

    for point in key_points:
        cv2.circle(test_pic, point, 5, (0, 255, 0), thickness=-1)

cv2.imshow('pic4', test_pic)
cv2.waitKey(0)