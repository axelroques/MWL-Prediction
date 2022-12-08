

import numpy as np


class HWeakClassifier():
    
    def __init__(self):

        self.tolerance = None

    def fit(self, X_train, y_train, tolerance=0.1):
        
        self.tolerance = tolerance
        
        self.threshold, self.value, self.side = self._find_threshold(X=X_train, y=y_train)

    def predict(self, X_test):

        y_predict = np.zeros(len(X_test))

        inds_right = (X_test >= self.threshold)

        y_predict[inds_right] = 1
            
        return y_predict
    
    def _find_threshold(self, X, y):
        
        self.proportion = np.sum(y==0)/len(y)

        self.threshold_left, self.value_left = self._find_threshold_one_side(X, y, "left")
        self.threshold_right, self.value_right = self._find_threshold_one_side(X, y, "right")

        if self.value_left >= self.value_right:
            
            side = "left"
            
            return self.threshold_left, self.value_left, side
        
        else:
            side = "right"

            return self.threshold_right, self.value_right, side

    def _find_threshold_one_side(self, X, y, side):
  
        tolerance=len(X)*self.tolerance
        
        if side=="left":
            weight_good=(1-self.proportion)/self.proportion
            weight_bad=1
            inds_sorted=np.argsort(X)
            class_inds_sorted=y[inds_sorted]

        else:
            weight_good=1
            weight_bad=(1-self.proportion)/self.proportion
            inds_sorted=np.argsort(X)[::-1]
            class_inds_sorted=(1-y)[inds_sorted]

        csum=np.cumsum(class_inds_sorted)

        nb_one_left = csum
        nb_zero_left = np.cumsum(1- class_inds_sorted)
        
        value_threshold = (nb_zero_left*weight_good-nb_one_left*weight_bad)[:-1]

        valid_inds = (csum <= (tolerance*weight_good)/weight_bad)[:-1] * (X[inds_sorted[1:]] != X[inds_sorted[:-1]])

        if np.sum(valid_inds)==0:
            return None, -np.inf

        ind_max_value=np.argmax(value_threshold*valid_inds)

        value=value_threshold[ind_max_value]
        
        threshold=np.mean([X[inds_sorted[ind_max_value]],X[inds_sorted[ind_max_value+1]]])

        return threshold, value



