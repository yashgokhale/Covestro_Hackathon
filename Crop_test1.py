
import tensorflow as tf
mnist = tf.keras.datasets.mnist
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np,copy,sys
from PIL import Image
import data_read as dr
import matplotlib.pyplot as plt
import pandas as pd


os.chdir("C:/Users/yashg/OneDrive/Documents/Covestro Hackathon/ParticleFormationStudentData/StudentData")

def cropped(Tabcsv,x):
    data=np.genfromtxt(Tabcsv,dtype=str,delimiter=',')
    data=data[1:]
    try:
        print(np.shape(data)[1])
        Image_Names=data[:,0]
        Rating=np.asarray(data[:,-1],dtype=float)/10
        Rating_=np.asarray(data[:,1:len(data[0])-2],dtype=float)
        return Image_Names
    except:
        Image_Names=copy.deepcopy(data)
        return Image_Names

    

def Matrisize(p):
    x=np.zeros([710,512])
    k=0
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j]=p[k]*255
            k+=1

    return x

Train='../StudentData/train_scores.csv'
Test='../StudentData/to_predict.csv'

Train_='../StudentData/train/'
Test_='../StudentData/test/'


x,y=dr.Reading(0)

Image_Names=cropped(Train,x)


#x_=dr.Reading(1)
Arr=[]
j=0
#for j in range(len(x)):
p=Matrisize(x[j])
#print(p)
#print(np.shape(p))
#np.savetxt('p.txt',p)
temp=copy.deepcopy(p)
k=0
for i in range(len(p[0])):
    if p[:,i].all()==0:
        #print(i)
        if k==0:
            index=i
            k=1
    else:
        temp[:,i]=p[:,i]
temp=temp[100:len(temp)-100,25:index-50]
img = Image.fromarray(temp)

#img.save(Train_ +Image_Names[j]+'.bmp')
#print(temp.shape)
Arr.append([temp.shape[0],temp.shape[1]])
print(j,Arr[j])
print(Arr)
Arr=np.array(Arr)
np.savetxt('Arr.txt',Arr,fmt='%.2f')

#print(np.where(p[:,:].all()==0))

img = Image.fromarray(temp)
img.show()

#np.savetxt('temp.txt',temp,fmt='%.2f')
#sys.exit()



cv2.imwrite("C:/Users/yashg/OneDrive/Documents/Covestro Hackathon/ParticleFormationStudentData/StudentData/", img.bmp)

