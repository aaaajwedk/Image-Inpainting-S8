import numpy as np
from numpy import asarray
import random
from PIL import Image
image = Image.open("D:/S7Project/dino.jpg")
arr = asarray(image)
for i in range(600):
    for j in range(600):
        sum=0
        for k in range(3):
            sum+=arr[i][j][k]
            arr[i][j]=sum
print(arr)
im = Image.fromarray(arr) #convert numpy array to image
im.save('test.jpg')