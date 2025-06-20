import cv2

image_path = "small-coco/train/train-000000191.png"
img = cv2.imread(image_path)
print(img.shape) 