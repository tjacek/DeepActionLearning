import os
import numpy as np
import seq.io,utils

def concat_files(paths,out_path='seqs/full'):
    if(os.path.isdir(paths)):
        paths=utils.bottom_dirs(paths)
    seq_type=get_seq_type(paths[0])
    read_actions=seq.io.build_action_reader(img_seq=seq_type,as_dict=True)
    action_sets=[read_actions(path_i) for path_i in paths]
    concated_actions=concat_actions(action_sets)
    save_actions=seq.io.ActionWriter(img_seq=seq_type)
    save_actions(concated_actions,out_path)

def concat_actions(action_sets):
    concated_actions=action_sets[0]
    if(len(action_sets)==1):
        return concated_actions
    for action_i in action_sets[1:]:
        concated_actions=pair_concat(concated_actions,action_i)
    return concated_actions

def pair_concat(actions1,actions2):
    names=actions2.keys()
    return { name_i:unify_actions(actions1[name_i],actions2[name_i],dim=1) 
                for name_i in names}

def unify_actions(action1,action2,dim=1):
    array1,array2=action1.as_array(),action2.as_array()
    if(array1.shape[0]!=array2.shape[0]):
        new_dim=min(array1.shape[0],array2.shape[0])
        array1,array2=array1[:new_dim],array2[:new_dim]
    new_array=np.concatenate((array1,array2),axis=dim)
    return action1.clone(new_array)

def get_seq_type(in_path):
    paths=utils.bottom_files(in_path)
    postfix=paths[0].split(".")[-1]
    return postfix=="png"