from segmentation import Unet
import cv2
import torch


file_path = 'data/CVPPPSegmData/data/A1/plant001_rgb.png'

img = cv2.imread(file_path)
img = cv2.resize(img, (64, 64))
img = img.transpose((2, 0, 1))

model = Unet()
model.load()
img = model.forward(torch.tensor([img]))
cv2.imshow("image", img)