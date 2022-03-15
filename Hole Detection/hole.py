import numpy as np
from numpy import asarray
import random
from PIL import Image
image = Image.open("D:\Project\Image-Inpainting-S8\Black Space Detection\dino.jpg")
# w, h = Image.size
arr = asarray(image)
row=0
blackpix=0
for i in range(600):
    if blackpix==96:
        row=i
        break
    blackpix=0
    for j in range(600):
        sum=0
        for k in range(3):
            sum+=arr[i][j][k]
            if sum==0:
                blackpix+=1
            else:
                pix=0
print(pix)
print(row)
# print(arr)
im = Image.fromarray(arr) #convert numpy array to image
im.save('test.jpg')