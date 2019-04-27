import numpy as np
import ensemble,utils#,ensemble.outliner

def all_feats_exper(feature_sets,arg_other,out_path="exp",ens=None):
    arg_other['dir_path']=feature_sets
    if(type(feature_sets)==str):
        feature_sets=utils.bottom_files(feature_sets)
        feature_sets=[ feature_i.split('/')[-1] for feature_i in feature_sets]
    arg_dicts=[]
    feature_sets+=[ feature_sets[:i] for i in range(1,len(feature_sets)+1)]
    for feature_set_i in feature_sets:
        dict_i=arg_other.copy()
        dict_i["common_paths"]=feature_set_i if(type(feature_set_i)==list ) else [feature_set_i]
        arg_dicts.append(dict_i)
    multi_experiment(arg_dicts,ens,out_path)

def multi_experiment(arg_dicts,ens=None,out_path="exp"):
    if(not ens):
        ens=ensemble.Ensemble()
    lines=[]
    for arg_dict_i in arg_dicts:
        lines+=simple_experiment(ens=ens,**arg_dict_i)
    to_csv(lines,out_path)

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

def to_csv(lines,out_path):
    out_file=open(out_path,'w+')
    out_file.write('\n'.join(lines))
    out_file.close()

def get_descritpion(deep_paths,feats):
    desc= "Yes" if(deep_paths) else "No"
    desc+= ",RFE" if(feats[0]) else ",None"
    return desc

def remove_sufix(name_i):
    return name_i.split('.')[0]

def subset_eperiment(arg_dict,clf_type,out_path,cat_subsets,legend=True):
    all_ensembles=[ensemble.Ensemble(clf_type,False,utils.CatSelector(a_i)) 
                        for a_i in cat_subsets]
    dir_path,common_paths,deep_paths,feats=arg_dict['dir'],arg_dict['common'],arg_dict['deep'],arg_dict['feats']
    lines=simple_experiment(dir_path,common_paths,deep_paths,feats,all_ensembles)
    to_csv(lines,out_path)

#def multi_experiment(arg_dicts,clf_type,out_path):
#    ens=ensemble.Ensemble(clf_type,False,None)
#    lines=[]
#    def dict_helper(arg_dict_i):
#        dir_path,common_paths,deep_paths,feats=arg_dict_i['dir'],arg_dict_i['common'],arg_dict_i['deep'],arg_dict_i['feats']
#        return simple_experiment(dir_path,common_paths,deep_paths,feats,ens)
#    for arg_dict_i in arg_dicts:
#        lines+=dict_helper(arg_dict_i)
#    to_csv(lines,out_path)

def weight_experiment(arg_dicts, clf_type,outliner_path):
    common_paths,deep_paths,feats=arg_dicts['common'],arg_dicts['deep'],arg_dicts['feats']
    weights=ensemble.outliner.get_weights(None,deep_paths,(0,0),outliner_path)
    ens=ensemble.WeightedEnsemble(weights,clf=clf_type)
    ens(common_paths,deep_paths,feats,show=True)

def single_exp(arg_dicts,clf_type,cf_path=True):
    common_paths,deep_paths,feats=arg_dicts['common'],arg_dicts['deep'],arg_dicts['feats']
    if( 'dir' in arg_dicts):
        dir_path=arg_dicts['dir']
        hc_paths=[ dir_path+'/'+path_i for path_i in common_paths]
    else:
        hc_paths=common_paths
    ens=ensemble.Ensemble(clf=clf_type)
    ens(hc_paths,deep_paths,feats,show=cf_path)

def dict_experiment(arg_dict,clf_type,out_path):
    ens=ensemble.Ensemble(clf_type,False,None)
    dir_path,common_paths,deep_paths,feats=arg_dict['dir'],arg_dict['common'],arg_dict['deep'],arg_dict['feats']
    lines=simple_experiment(dir_path,common_paths,deep_paths,feats,ens)
    to_csv(lines,out_path)





