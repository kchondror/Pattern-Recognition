# -*- coding: utf-8 -*-

import numpy as np
import random as rd


class Kmeans:
    
    
    def __init__(self,M):
        self.M=M
        self.centroids=[]
        self.numOfCenters=0
        
        
    def getCentroids(self):
        return self.centroids
              
                      
    def MaxiMin(self):
        self.centroids.append(rd.choice(self.M))#Αρχικά επιλεγεί ενα τυχαίο διάνυσμα.
        self.numOfCenters+=1
        
        #Βρίσκει το διάνυσμα που βρίσκεται πιο μακριά από το προηγούμενο κέντρο.
        maxDist=-1
        for i in self.M:
            j=self.centroids[self.numOfCenters-1]
            dist=np.linalg.norm(i.vector-j.vector)
            if(dist > maxDist):
               maxDist=dist
               maxDistVector=i

        self.centroids.append(maxDistVector)
        self.numOfCenters+=1
    
        #Βρίσκει το μέγιστο από τις ελάχιστες αποστάσεις των υπολοίπων διανυσμάτων 
        #απτα 2 προηγούμενα κέντρα.
        while(self.numOfCenters<4):
            minDist={}
            
            for i in self.M:
                mins=[]
                for j in self.centroids:
                    mins.append(np.linalg.norm(i.vector-j.vector))
            
                minDist[min(mins)]=i
            
            
            self.centroids.append(minDist.get(max(minDist.keys())))
            self.numOfCenters+=1
        
         
    
    def KMeansAlgorithm(self):
        
        self.MaxiMin()
        
        
        beforeUpdate=[]
        while(True):
                
            #Υπολογίζει την απόσταση του κάθε διανύσματος από όλα τα κέντρα 
            #και το προσθέτει στην ομάδα του πιο κοντινού.
            for i in self.M:
                Min=float('inf')
                for j in self.centroids:
                    dist=np.linalg.norm(i.vector-j.vector)
                    if(dist<Min):
                        Min=dist
                        nearestCenter=j
                
                nearestCenter.addSubVector(i)
            
            
            #Ενημερώνει την τιμή των κέντρων.
            count=0
            for i in self.centroids:
                v=i.vector
                beforeUpdate.append(v)
                i.UpdateCentroid()
                count+=1
            
            #Ελέγχει αν τα κέντρα έχουν την ιδιά τιμή με αυτή της προηγουμένης  
            #επανάληψης και αν ναι τερματίζει ο αλγόριθμος.
            count=0
            for i in self.centroids:
                for j in beforeUpdate:
                    if(np.array_equal(i.vector,j)):
                        count+=1
                        break
            
            if(count==4):
                break
            else:
                for i in self.centroids:
                    i.subVectors=[]
        
        
            
            
                
    
        
    