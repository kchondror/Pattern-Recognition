# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import gzip

from KMeansClass import Kmeans
from VectorClass import Vector
from PCAClass import PCA
#------------------------------------------------------------------------------------------

#Κάνει plot τις εικόνες από το αρχικό dataset.
def Plot(data):
    image = np.asarray(data).squeeze()
    plt.imshow(image)
    plt.show()


#Διαβάζει τα αρχεία και επιστρέφει πίνακες με τις εικόνες και τα labels.
def LoadData(imageFile,labelFile,Filesize):
    
    imageFile = gzip.open(imageFile,'r')
    imageFile.read(16)
    
    labelFile = gzip.open(labelFile,'r')
    labelFile.read(8)
    
    count=0
    LabelArray=[]
    ImageArray=[]
    
    while(True):
        
        if(count==Filesize):break 
        
        #Παίρνει τα επόμενα labels - images σε κάθε loop.
        label=np.frombuffer(labelFile.read(1), dtype=np.uint8).astype(np.int64)
        
        image = np.frombuffer(imageFile.read(28 * 28), dtype=np.uint8).astype(np.float32)
        
        #Φιλτράρει τις εικόνες.
        if(label[0] in [1,3,7,9]):
            
            ImageArray.append(image.reshape(28,28))
            LabelArray.append(label[0])
            
        count+=1
        
    return ImageArray,LabelArray
    
#Δημιουργεί ενα scatterPlot για τα σημεία και για τα κέντρα των ομάδων.
def PlotScatter(points,centroids):
    
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    
    x=[]
    y=[]
    z=[]
    for i in points:
        x.append(i.vector[0])
        y.append(i.vector[1])
        z.append(i.color)
        
    data=np.array([[x],[y]])
    colorss=np.array([z])
    
    
    plt.scatter(data[0][:],data[1][:],c=colorss[0][:])
    
    #Τα χρώματα μπαίνουν τυχαία σε κάθε ομάδα .
    colors=["red","blue","green","yellow"]
    count=0
    
    #Τρέχει οταν κανουμε pass ομάδες-κέντρα.
    for i in centroids:
        x=[]
        y=[]
        
        for j in i.subVectors:
            x.append(j.vector[0])
            y.append(j.vector[1])
           
        data=np.array([[x],[y]])
        plt.scatter(data[0][:],data[1][:],c=colors[count])
        
        plt.scatter(i.vector[0], i.vector[1], c ="black",
                    linewidths = 2,
                    marker ="^",
                    edgecolor ="pink",
                    s = 200)
        count+=1
        
    
    plt.show()


def DimensionRed(m,NumOfDim):
    
    
    dimReduction=PCA(m,NumOfDim)
    m=dimReduction.PCA()#Επιστρέφει τον πίνακα με τις μειωμένες διαστάσεις.

    #Δημιουργεί τα διανύσματα
    M3=[]
    count=0
    for i in m:
        vec=Vector(i,Ltr[count])
        count+=1
        M3.append(vec)


    means=Kmeans(M3)
    means.KMeansAlgorithm()

    #Κάνει Plot το σχήμα για τα δεδομένα με 2 διαστάσεις.
    if(NumOfDim==2):
        PlotScatter(M3,[])
        PlotScatter([], means.getCentroids())


    #Υπολογίζει το average του purity όλων των κλάσεων.
    Sum=0
    for i in means.centroids:
        Sum+=i.purityScore()
    
    total=Sum/len(means.centroids)
    print("Exercise 4 (V={:d}) purity :{:f}".format(NumOfDim,total))
    return total



#------------------------------------main--------------------------------------------------

#Ex.1
N,Lte=LoadData('test_images.gz', 'test_labels.gz', 10000)

M,Ltr=LoadData('train_images.gz', 'train_labels.gz',60000)

#Plot(M[0])
#print(Ltr[0])

#Ex.2
M=np.asarray(M)

M2=[]#Πίνακας με αντικείμενα της vectorClass.
for matrix,label in zip(M,Ltr):
    
    rows=[]
    columns=[]
    for i in range(28): 
        if i % 2 == 1:           
            rows.append(matrix[i][:])        
        else:
            columns.append(matrix[:][i])
      
    vec=np.array([np.mean(rows),np.mean(columns)])
    M2.append(Vector(vec,label))   
    

PlotScatter(M2, [])

#Ex.3

means = Kmeans(M2)
means.KMeansAlgorithm()

PlotScatter([], means.centroids)


#Υπολογίζει το average του purity όλων των κλάσεων.
Sum=0
for i in means.centroids:
    Sum+=i.purityScore()

print("Exercise 3 purity : {:f}".format(Sum/len(means.centroids)))



#Ex.4

#Κάνει reshape τα δεδομένα ώστε να μπορεί να εφαρμοστεί ο PCA αλγόριθμος σε αυτά.
m=np.asarray(M)
m=np.reshape(m,(m.shape[0],784))

#Εμφανίζει το purity για τις ανάλογες διστάσεις.
V=[2,25,50,100]
maxPurity=[]
for i in V:
    maxPurity.append(DimensionRed(m,i))

print("Max Purity (V={:d}) : {:f}".format(V[maxPurity.index(max(maxPurity))],max(maxPurity)))


















