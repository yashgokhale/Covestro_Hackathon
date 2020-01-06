
import numpy as np,copy,sys
from PIL import Image

from multiprocessing import Pool

def Reading(T):
        
    Train='../StudentData/train_scores.csv'
    Test='../StudentData/to_predict.csv'

    Location1='../StudentData/train/'
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
            image= Image.open(Loc+Image_Names[i]+'.bmp','r')
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
