import numpy as np
from sklearn.metrics import accuracy_score

def cls_accuracy(y_true,all_pred,show=True,stats=True):
    indvidual_acc=[accuracy_score(y_true,pred_i)
                    for pred_i in all_pred]
    if(show):
        for i,indiv_i in enumerate(indvidual_acc):
    	    print("%d %f" % (i,indiv_i))
    if(stats):
        return basic_stats(indvidual_acc)
    else:
        return indvidual_acc

def basic_stats(array):
    return np.amin(array),np.median(array),np.mean(array),np.amax(array)