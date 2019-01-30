import numpy as np
from collections import Counter
import basic,utils
import ensemble.tools,ensemble.inspect

def learning(handcrafted_path=None,deep_path=None,feats=(250,100)):
    datasets_dict=get_datasets(handcrafted_path,deep_path)
    feat_reduction(datasets_dict,hc_feats=feats[0],deep_feats=feats[1])
    datasets=preproc_dataset(datasets_dict)
    y_true,all_pred=get_prediction(datasets)
    indiv=ensemble.inspect.cls_accuracy(y_true,all_pred,stats=False)
    y_pred=vote(all_pred)
    ensemble.tools.show_result(y_true,y_pred,datasets[0])
    ensemble.tools.show_stats(y_true,y_pred,datasets[0])

def get_datasets(handcrafted_path=None,deep_path=None):
    if(not handcrafted_path and not deep_path):
        raise Exception("No dataset paths")
    handcrafted_dataset,deep_datasets=None,None
    if(handcrafted_path):
        handcrafted_dataset= read_data(handcrafted_path)
    if(deep_path):
        deep_paths=utils.bottom_files(deep_path)
        if(len(deep_paths)==0):
            raise Exception("No datasets at " + deep_path)
        deep_datasets=[read_data(path_i) for path_i in deep_paths]
    return {"handcrafted":handcrafted_dataset,"deep":deep_datasets}

def feat_reduction(datasets_dict,hc_feats=250,deep_feats=100):
    hc_data=datasets_dict['handcrafted']
    if(hc_data):
        datasets_dict['handcrafted']=ensemble.tools.rfe_selection(hc_data,hc_feats)
    if(datasets_dict['deep']):
        datasets_dict['deep']=[ ensemble.tools.rfe_selection(deep_i,deep_feats)  
                                    for deep_i in datasets_dict['deep']]
    return datasets_dict

def preproc_dataset(datasets_dict):
    datasets=datasets_dict["deep"]
    if(datasets_dict["handcrafted"]):
        hc_data=datasets_dict["handcrafted"]
        if(datasets_dict["deep"]):
            datasets=[basic.unify_datasets([hc_data,deep_i])
                        for deep_i in datasets]
        else:
            datasets=[hc_data]
    print("Number of feats %d " % datasets[0].dim())
    return datasets
   
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

def read_data(in_path,norm=True):
    dataset=basic.read_dataset(in_path)
    if(norm):
        dataset.norm()
    return dataset