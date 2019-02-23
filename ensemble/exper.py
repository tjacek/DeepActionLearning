import ensemble

def multi_experiment(arg_dicts,clf_type,out_path):
    ens=ensemble.Ensemble(clf_type,False,None)
    lines=[]
    def dict_helper(arg_dict_i):
        dir_path,common_paths,deep_paths,feats=arg_dict_i['dir'],arg_dict_i['common'],arg_dict_i['deep'],arg_dict_i['feats']
        return simple_experiment(dir_path,common_paths,deep_paths,feats,ens)
    for arg_dict_i in arg_dicts:
        lines+=dict_helper(arg_dict_i):
    to_csv(lines,out_path)

def dict_experiment(arg_dict,clf_type,out_path):
    ens=ensemble.Ensemble(clf_type,False,None)
    dir_path,common_paths,deep_paths,feats=arg_dict['dir'],arg_dict['common'],arg_dict['deep'],arg_dict['feats']
    lines=simple_experiment(dir_path,common_paths,deep_paths,feats,ens)
    to_csv(lines,out_path)

def simple_experiment(dir_path,common_paths,deep_paths,feats,ensemble):
    desc=get_descritpion(deep_paths,feats)
    def line_helper(feature_set_i):
        if( not feature_set_i and not deep_paths):
            return ''
        name_i=get_name(feature_set_i)
        print(name_i)
        full_path_i=get_full_paths(dir_path,feature_set_i)
        score_i,n_feats_i=ensemble(full_path_i,deep_paths,feats,show=False)
        return '%s,%s,%d,%s' % (name_i,desc,n_feats_i,score_i)
    return [line_helper(feature_set_i) for feature_set_i in common_paths]      

def to_csv(lines,out_path):
    out_file=open(out_path,'w+')
    out_file.write('\n'.join(lines))
    out_file.close()

def get_descritpion(deep_paths,feats):
    desc= "Yes" if(deep_paths) else "No"
    desc+= ",RFE" if(feats[0]) else ",None"
    return desc

def get_name(feature_set_i):
    if(type(feature_set_i)==str):
        return remove_sufix(feature_set_i)
    if(type(feature_set_i)==list):
        feature_set_i=[ remove_sufix(name_i) for name_i in  feature_set_i]
        return '+'.join(feature_set_i)
    return "-"

def get_full_paths(dir_path,feature_set_i):
    if(type(feature_set_i)==str):
        return dir_path+'/'+feature_set_i
    if(type(feature_set_i)==list):
        return [dir_path+'/'+feature_i for feature_i in feature_set_i]
    return []

def remove_sufix(name_i):
    return name_i.split('.')[0]