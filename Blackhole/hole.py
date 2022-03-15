#import packages
import os
import numpy as np
from PIL import Image
from numpy import asarray
import random

path="./data"
dir_list = os.listdir(path)
print(dir_list)
#Open single image and convert to array
for k in dir_list:
    img=Image.open("./data/"+k)
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
    im.save("./hole/"+k)

