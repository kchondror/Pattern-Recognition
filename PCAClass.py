# -*- coding: utf-8 -*-

import numpy as np

class PCA:
    
    def __init__(self,X,V):
        
        self.Dataset=X
        self.NumOfDim=V
              
        
    def PCA(self):
        
        #Υπολογίζει τον πίνακα συνδιακύμανσης.
        matrix = np.cov(self.Dataset , rowvar = False)
         
        #Βρίσκει τις ιδιοτιμές και τα ιδιοδιανύσματα του.
        w , v = np.linalg.eigh(matrix)
        
        #Κάνει sort τον πίνακα των ιδιοδιανυσμάτων με βάση τις μέγιστες ιδιοτίμες.
        v = v[:,np.argsort(w)[::-1]]
        
        #Παίρνει μόνο τον αριθμό των διαστάσεων που με ενδιαφέρει .
        v = v[:,0:self.NumOfDim]
        
        #Βρίσκει το τελικό dataSet με μειωμένες διαστάσεις.
        finalData = np.dot(v.transpose() , self.Dataset.transpose() ).transpose()
        
        return finalData




