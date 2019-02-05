import numpy as np
from collections import Counter
import ensemble.data
import ensemble.tools,ensemble.inspect

class Ensemble(object):
    def __init__(self,get_model=None):
        self.get_model=get_model if(get_model) else tools.train_model

def learning(handcrafted_path=None,deep_path=None,feats=(250,100)):
    datasets=ensemble.data.get_datasets(handcrafted_path,deep_path,feats)
    y_true,all_pred=get_prediction(datasets)
    indiv=ensemble.inspect.cls_accuracy(y_true,all_pred,stats=False)
    y_pred=vote(all_pred)
    ensemble.tools.show_result(y_true,y_pred,datasets[0])
    ensemble.tools.show_stats(y_pred)
   
def get_prediction(datasets):
    result=[ ensemble.tools.train_model(i,data_i) 
                    for i,data_i in enumerate(datasets)]
    y_true=result[0][0]
    all_pred=[result_i[1] for result_i in result]
    return y_true,all_pred

def vote(all_votes):
    all_votes=np.array(all_votes)
    y_pred=[]
    for vote_i in all_votes.T:
        count =Counter(vote_i)
        cat_i=count.most_common()[0][0]
        y_pred.append(cat_i)
    return y_pred