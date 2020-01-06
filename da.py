import numpy as np, sys
from scipy import ndimage    
from PIL import Image
from multiprocessing import Pool
import matplotlib.pyplot as plt
#from keras.models import Sequential
#model = Sequential()


import csv

Train='../StudentData/train_scores.csv'
Location='../StudentData/train/'

data=np.genfromtxt(Train,dtype=str,delimiter=',')
data=data[1:]
Image_Names=data[:,0]
Rating=np.asarray(data[:,-1],dtype=float)

def read_data(i):
    image= Image.open(Location+Image_Names[i]+'.bmp','r')
    pix_val = list(image.getdata())
    return pix_val

def Matrisize(p):
    x=np.zeros([710,512])
    k=0
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j]=p[k]
            k+=1

    return x

x=range(4)
print(x)



temp=read_data(0)
plt.plot(temp,lw=0.1)
plt.show()
pixels=[]
for i in range(len(Image_Names)):
    image= Image.open(Location+Image_Names[i]+'.bmp','r')
    pix_val = list(image.getdata())
    pixels.append(pix_val)

pixels=np.asarray(pixels)

sys.exit()

from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([Dense(32, input_shape=(len(pixels[0]),)),
Activation('relu'),Dense(10),Activation('softmax'),
])


model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))


sys.exit()
#'''
y=Pool()
pixels=y.map(read_data,x)
y.close()
#y.join()
#'''



sys.exit()
print(Image_Names)

import os
from scipy import misc









print(np.shape(pix_val))
print(pix_val[0])
#print(pix_val)
