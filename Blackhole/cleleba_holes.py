#import packages
import os
import numpy as np
from PIL import Image
from numpy import asarray
import random
from threading import Thread
from time import perf_counter


path="./data"
dir_list = os.listdir(path)

def task():
    
#print(dir_list)
#Open single image and convert to array
    r=0
    while(r<(l/2)):
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
            im.save("./holes/"+k)
        r=r+1
def task2():
    
#print(dir_list)
#Open single image and convert to array
    p=(l/2)
    while((p<l)):
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
            im.save("./holes/"+k)            

l=len(dir_list)
#print(l)
s="./data"
v="./data_2"
i=0
threads = []
# while(i<(l/2)):
#     for n in dir_list:
t1 = Thread(target=task)
t2 = Thread(target=task)
t1.start()
t2.start()

# wait for the threads to complete
t1.join()
t2.join()

# t1 = Thread(target=task, args=(s,))
# t2 = Thread(target=task,args=(v))

# # start the threads
# t1.start()
# t2.start()

# # wait for the threads to complete
# t1.join()
# t2.join()


