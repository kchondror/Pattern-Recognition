# -*- coding: utf-8 -*-

import numpy as np

class Vector:
    
    #Οι απαραίτητες πληροφορίες που θέλουμε να έχει εάν διάνυσμα.
    def __init__(self,vector,label):
        self.vector=vector
        
        self.label=label
        self.color=self.setColor(label)
        self.subVectors=[]
        
        
    #Ανάλογα το label εκχωρείτε και το αντίστοιχο χρώμα.
    def setColor(self,label):
        
        if(label==1):
            color="red"
        elif(label==3):
            color="green"
        elif(label==7):
            color="blue"
        else:
            color="yellow"
        
        return color
            
    
    def addSubVector(self,vec):
        self.subVectors.append(vec)
        
        
    #Ενημερώνει το κέντρο με βάση το mean των διανυσμάτων που έχει ως sub-vectors.
    def UpdateCentroid(self):

        sumVec=np.zeros(self.vector.shape)
        for i in self.subVectors:
            sumVec=np.add(sumVec,i.vector)
        
        self.vector=sumVec/len(self.subVectors)
            
            
    #Υπολογίζει το purity του κέντρου.    
    #Πηγή: https://towardsdatascience.com/evaluation-metrics-for-clustering-models-5dde821dd6cd
    def purityScore(self):
        
        #Βρίσκει το label με τις περισσότερες εμφανίσεις στον πίνακα subvector
        #ώστε να είναι αυτό το οποίο αντιπροσωπεύει την ομάδα.
        mins={1:0,3:0,7:0,9:0}
        for i in self.subVectors:
            mins[i.label]+=1
        
        MaxLabel = max(mins, key=mins.get)
        true=0

        for i in self.subVectors:
            if(i.label==MaxLabel):
                true+=1

    
        return true/len(self.subVectors)
            
            
            

            