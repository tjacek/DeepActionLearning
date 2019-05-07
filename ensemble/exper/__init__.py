import numpy as np
import ensemble,utils,ensemble.tools

def build_ensemble(dataset="?",clf_type="SVC",selector=None):
    clf=ensemble.tools.SVC_cls if(clf_type=="SVC") else None
    ens=ensemble.Ensemble(clf,selector=selector)
    ens_name= dataset+"_"+clf_type+".csv"
    return clf,ens_name

def all_feats_exper(feature_sets,arg_other,out_path="exp",ens=None):
    arg_other['dir_path']=feature_sets
    if(type(feature_sets)==str):
        feature_sets=utils.bottom_files(feature_sets)
        feature_sets=[ feature_i.split('/')[-1] for feature_i in feature_sets]
    feature_sets+=[ feature_sets[:i] for i in range(1,len(feature_sets)+1)]
    print(feature_sets)
    arg_other["common_paths"]=feature_sets
    multi_experiment([arg_other],ens,out_path)

def multi_experiment(arg_dicts,ens=None,out_path="exp"):
    if(not ens):
        ens=ensemble.Ensemble()
    lines=[]
    for arg_dict_i in arg_dicts:
        lines+=simple_experiment(ens=ens,**arg_dict_i)
    to_csv(lines,out_path)

def subset_eperiment(arg_dict,clf_type,out_path,cat_subsets,legend=True):
    all_ensembles=[ensemble.Ensemble(clf_type,False,utils.CatSelector(a_i)) 
                        for a_i in cat_subsets]
    dir_path,common_paths,deep_paths,feats=arg_dict['dir'],arg_dict['common'],arg_dict['deep'],arg_dict['feats']
    lines=simple_experiment(dir_path,common_paths,deep_paths,feats,all_ensembles)
    to_csv(lines,out_path)

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