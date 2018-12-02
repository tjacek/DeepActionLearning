import numpy as np
import seq.io

def concat(in_path1='seqs/max_z',in_path2='seqs/all',out_path='seqs/full'):
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=True)
    actions1,actions2=read_actions(in_path1),read_actions(in_path2)
    names=actions1.keys()
    unified_actions=[ unify_actions(actions1[name_i],actions2[name_i],dim=1) 
                        for name_i in names]
    save_actions=seq.io.ActionWriter(img_seq=False)
    save_actions(unified_actions,out_path)

def unify_actions(action1,action2,dim=1):
    array1,array2=action1.as_array(),action2.as_array()
    if(array1.shape[0]!=array2.shape[0]):
        new_dim=min(array1.shape[0],array2.shape[0])
        array1,array2=array1[:new_dim],array2[:new_dim]
    print(array1.shape)
    print(array2.shape)
    new_array=np.concatenate((array1,array2),axis=dim)
    print(new_array.shape)
    return action1.clone(new_array)