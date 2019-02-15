def simple_experiment(dir_path,common_paths,deep_paths,feats,ensemble):
    result=''
    for feature_set_i in common_paths:
        name_i=get_name(feature_set_i)
        print(name_i)
        full_path_i=get_full_paths(dir_path,feature_set_i)
        line_i=ensemble(full_path_i,deep_paths,feats)
        result+=name_i+','+line_i+'\n'
    print(result)

def get_name(feature_set_i):
    if(type(feature_set_i)==str):
        return feature_set_i
    if(type(feature_set_i)==list):
        return '+'.join(feature_set_i)
    return '-'

def get_full_paths(dir_path,feature_set_i):
    if(type(feature_set_i)==str):
        return dir_path+'/'+feature_set_i
    if(type(feature_set_i)==list):
        return [dir_path+'/'+feature_i for feature_i in feature_set_i]
    return []


