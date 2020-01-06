
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np,copy,sys
from PIL import Image
import os
#print(os.getcwd())
os.chdir("C:/Users/yashg/OneDrive/Documents/Covestro Hackathon/ParticleFormationStudentData/StudentData")
def read_data(i):
    image= Image.open(Location+Image_Names[i]+'.bmp','r')
    pix_val = list(image.getdata())
    return np.array(pix_val)

# load dataset

def Matrisize(p):
    x=np.zeros([710,512])
    k=0
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j]=p[k]
            k+=1

    return x


Train='train_scores.csv'
Location='im_crop/'

data=np.genfromtxt(Train,dtype=str,delimiter=',')
data=data[1:]
Image_Names=data[:,0]
Rating=np.asarray(data[:,-1],dtype=float)
Rating_=np.asarray(data[:,1:len(data[0])-2],dtype=float)

#'''
Y=[]
temp=np.zeros(100)

for i in range(len(Rating)):
    temp1=int(10*Rating[i])
    temp[temp1]=1
    Y.append(temp)
Y=np.array(Y)
#'''
pixels=[]
for i in range(len(Image_Names)):
    image= Image.open(Location+Image_Names[i]+'.bmp','r')
    pix_val = list(image.getdata())
    p=Matrisize(pix_val)
    #print(p.shape)
    pixels.append(p)
X=np.asarray(pixels)/255.0


print(np.shape(X))
print(np.shape(Y))
nx=len(X[0])
ny=len(Y[0])





# Install TensorFlow

import tensorflow as tf
mnist = tf.keras.datasets.mnist
from tensorflow import keras
from tensorflow.keras import layers





#(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train, x_test = x_train / 255.0, x_test / 255.0

x_train=X
y_train=np.array(Rating)





def build_model():
    model = keras.Sequential([
    layers.Flatten(input_shape=X[0].shape),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model


model = build_model()
model.summary

model.fit(x_train,y_train,epochs=100)

Test='to_predict.csv'
Location2='test/'
data2=np.genfromtxt(Test,dtype=str,delimiter=',')
#data2=data2[1:]
Image_Names2=data2[:,0]

pixels2=[]
for i in range(len(Image_Names2)):
    image2= Image.open(Location2+Image_Names2[i]+'.bmp','r')
    pix_val2 = list(image2.getdata())
    p2=Matrisize(pix2_val)
    #print(p.shape)
    pixels.append(p2)
X2=np.asarray(pixels2)/255.0

x_test=X2

model.predict(x_test)

