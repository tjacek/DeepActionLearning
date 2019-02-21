def simple_experiment(dir_path,common_paths,deep_paths,feats,ensemble,out_path='result.csv'):
    result=''
    desc=get_descritpion(deep_paths,feats)
    for feature_set_i in common_paths:
        name_i=get_name(feature_set_i)
        print(name_i)
        full_path_i=get_full_paths(dir_path,feature_set_i)
        score_i,n_feats_i=ensemble(full_path_i,deep_paths,feats,show=False)
        result+='%s,%s,%d,%s\n' % (name_i,desc,n_feats_i,score_i)
    out_file=open(out_path,'w+')
    out_file.write(result)
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