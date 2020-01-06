import numpy as np,copy,sys
from PIL import Image
import os
from multiprocessing import Pool

os.chdir("C:/Users/yashg/OneDrive/Documents/Covestro Hackathon/ParticleFormationStudentData/StudentData")

def Reading(T):
        
    Train='../StudentData/train_scores.csv'
    Test='../StudentData/to_predict.csv'

    Location1='../StudentData/CROPPED_TRAINCASES/'
    Location2='../StudentData/test/'

    def data(Tabcsv,Loc):
        data=np.genfromtxt(Tabcsv,dtype=str,delimiter=',')
        data=data[1:]
        try:
            print(np.shape(data)[1])
            Image_Names=data[:,0]
            Rating=np.asarray(data[:,-1],dtype=float)/10
            Rating_=np.asarray(data[:,1:len(data[0])-2],dtype=float)
        except:
            Image_Names=copy.deepcopy(data)
        

        def read_data(i):
            image= Image.open(Loc+Image_Names[i]+'.jpg','r')
            pix_val = list(image.getdata())
            return np.array(pix_val)

        pixs=[]
        for i in range(len(Image_Names)):
            pix=read_data(i)
            pixs.append(pix)

        pixs=np.array(pixs)/255.0
        print(np.shape(pixs))

        try:
            return pixs,Rating
        except:
            return pixs
        
    if T==0:
        X,Y=data(Train,Location1)
        return X,Y
    if T==1:
        X=data(Test,Location2)
        return X

#Reading(0)[1]
import matplotlib.pyplot as plt

#for i in range(1):
    #for j in range(196):
        #if a[i,j]>0.5:
            #n[j]=n[j]+1
pixels,rating=np.array(Reading(0))
n=np.zeros(197)
n.shape
for i in range(len(n)):
    for j in range(363519):
        if pixels[i][j]>0.9:
            n[i]=n[i]+1

n_train=np.array(n[0:160])
n_test=np.array(n[161:])
import pandas as pd
train_data=pd.read_csv("train_scores.csv")

mavg=train_data["total_rating"]
mavg=np.array(mavg)

m_train=np.array(mavg[0:160])
m_test=n_test=np.array(mavg[161:])

X = np.column_stack([n_train**3,n_train**2,n_train, n_train**0])

#  I find these intermediate variables make it easier to read
XTX = np.dot(X.T, X)
XTy = np.dot(X.T, m_train)

a,b,c,d= np.dot(np.linalg.inv(XTX), XTy)

y_pred=a*m_test**3+b*m_test**2+c*m_test+d

R2=1-(np.sum(m_test-y_pred)**2/np.sum(m_test-np.mean(m_test)))

#print(R2)

plt.plot(n,mavg,'.')