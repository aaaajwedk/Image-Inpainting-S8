#import packages
import numpy as np
from PIL import Image
from numpy import asarray
import random

#Open single image and convert to array
img=Image.open("one.jpg")
w,h=img.size
data1=asarray(img)

#random
r=random.randrange(w-96)
r1=random.randrange(h-96)

#Add holes in the image
for i in range(r1,r1+96):
    for j in range(r,r+96):
        data1[i][j]=[0,0,0]
im = Image.fromarray(data1) #convert numpy array to image
im.save('converted.jpg')

