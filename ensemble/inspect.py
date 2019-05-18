import numpy as np
from sklearn.metrics import accuracy_score
import ensemble,ensemble.tools

def ensemble_matrix(arg_dicts,test_ensemble=None):
    y_true,all_pred=get_preds(arg_dicts,test_ensemble)
    all_pred_by_cat=[by_cat(y_true,pred_i) for pred_i in all_pred]
    acc_by_cat=[class_accuracy(pred_i) for pred_i in all_pred_by_cat]
    acc_by_cat=np.array(acc_by_cat)
    ensemble.tools.heat_map(acc_by_cat)

def ensemble_stas(arg_dicts,test_ensemble=None):
    y_true,all_pred=get_preds(arg_dicts,test_ensemble)
    clfs_acc=clfs_accuracy(y_true,all_pred)
    print(clfs_acc)
    print("min %f median %f mean %f max %f" % basic_stats(clfs_acc))

def get_preds(arg_dicts,test_ensemble=None):
    test_ensemble= test_ensemble if(test_ensemble) else ensemble.Ensemble()
    datasets,n_feats=test_ensemble.get_datasets(**arg_dicts)
    y_true,all_pred=test_ensemble.get_prediction(datasets)
    return y_true,indiv_votes(all_pred)

def clfs_accuracy(y_true,all_pred):
    return [accuracy_score(y_true,pred_i)
                    for pred_i in all_pred]

def by_cat(y_true,y_pred):
    n_cats=max(y_true)
    pred_by_cat=[[] for i in range(n_cats)]
    for true_i,pred_i in zip(y_true,y_pred):
        pred_by_cat[true_i-1].append(pred_i)
    return pred_by_cat

def class_accuracy(y_pred_by_cat):
    def acc_helper(i,cat_i):
        cat_i=np.array(cat_i,dtype=float)
        cat_i[cat_i!=i]=0.0
        cat_i[cat_i==i]=1.0
        return np.mean(cat_i)
    class_acc=[acc_helper(i+1,cat_i) for i,cat_i in enumerate(y_pred_by_cat)]
    return np.round(class_acc,2) 

def basic_stats(array):
    return np.amin(array),np.median(array),np.mean(array),np.amax(array)

def indiv_votes(raw_votes):
    raw_votes=np.transpose(raw_votes,axes=(1,0,2))
    indiv_acc=[[np.argmax(vote_ij)+1  
                    for vote_ij in vote_i] 
                        for vote_i in raw_votes]
    indiv_acc=np.array(indiv_acc).T
    return indiv_acc