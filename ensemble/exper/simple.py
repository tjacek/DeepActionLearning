import numpy as np

def simple_experiment(dir_path,common_paths,deep_paths,feats,ens,legend=True):
    desc=get_descritpion(deep_paths,feats)
    if(legend):
        lines=["Feature Sets,Deep Ensemble,Feature Selection,Number of Features,Accuracy,Precision,Recall,F1-score"]
    else:
        lines=[]
    if(not deep_paths):
        common_paths=[feature_set_i for feature_set_i in common_paths if(feature_set_i)]    
    def line_helper(feature_set_i):
        name_i=get_name(feature_set_i)
        print(name_i)
        full_path_i=get_full_paths(dir_path,feature_set_i)
        if(type(ens)==list):
            score_i,n_feats_i=get_avg_ensemble(full_path_i,deep_paths,feats,ensemble)
        else:
            score_i,n_feats_i=ens(full_path_i,deep_paths,feats,show=False)
            print(score_i)
        return '%s,%s,%d,%s' % (name_i,desc,n_feats_i,score_i)
    return lines+[line_helper(feature_set_i) for feature_set_i in common_paths]

def get_avg_ensemble(full_path_i,deep_paths,feats,all_ensemble):
    metrics=[ensemble_i(full_path_i,deep_paths,feats,show=False) for ensemble_i in all_ensemble]
    n_feats=metrics[0][1]
    metrics=np.array([metric_i[0] for metric_i in metrics])
    metrics=np.mean(metrics,axis=0)
    metrics= "%.4f,%.4f,%.4f,%.4f," % (metrics[0],metrics[1],metrics[2],metrics[3])
    return metrics,n_feats  

def get_full_paths(dir_path,feature_set_i):
    if(type(feature_set_i)==str):
        return dir_path+'/'+feature_set_i
    if(type(feature_set_i)==list):
        return [dir_path+'/'+feature_i for feature_i in feature_set_i]
    return []

def get_name(feature_set_i):
    if(type(feature_set_i)==str):
        return remove_sufix(feature_set_i)
    if(type(feature_set_i)==list):
        feature_set_i=[ remove_sufix(name_i) for name_i in  feature_set_i]
        return '+'.join(feature_set_i)
    return "-"

def get_descritpion(deep_paths,feats):
    desc= "Yes" if(deep_paths) else "No"
    desc+= ",RFE" if(feats[0]) else ",None"
    return desc

def remove_sufix(name_i):
    return name_i.split('.')[0]