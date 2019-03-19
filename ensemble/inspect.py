import numpy as np
from sklearn.metrics import accuracy_score
import ensemble

def ensemble_stast(arg_dicts,test_ensemble=None):
    test_ensemble= test_ensemble if(test_ensemble) else ensemble.Ensemble()
    common_paths,deep_paths,feats=arg_dicts['common'],arg_dicts['deep'],arg_dicts['feats']
    datasets,n_feats=test_ensemble.get_datasets(common_paths,deep_paths,feats)
    y_true,all_pred=test_ensemble.get_prediction(datasets)
    all_pred=indiv_votes(all_pred)
    indiv_acc=individual_accuracy(y_true,all_pred)
    print(indiv_acc)
    print("min %f median %f mean %f max %f" % basic_stats(indiv_acc))

def individual_accuracy(y_true,all_pred):
    return [accuracy_score(y_true,pred_i)
                    for pred_i in all_pred]

def basic_stats(array):
    return np.amin(array),np.median(array),np.mean(array),np.amax(array)

def indiv_votes(raw_votes):
    raw_votes=np.transpose(raw_votes,axes=(1,0,2))
    indiv_acc=[[np.argmax(vote_ij)+1  
                    for vote_ij in vote_i] 
                        for vote_i in raw_votes]
    indiv_acc=np.array(indiv_acc).T
    return indiv_acc