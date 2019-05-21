import numpy as np
from sets import Set
import ensemble.data,ensemble.pick
import basic,utils

def pairwise_concat(args_dict,detector_path,out_path):
    sep_matrix=ensemble.pick.feats_inliners(args_dict,detector_path)
    feat_pairs=get_pairs(sep_matrix)
    concat_helper= make_concat_helper(args_dict)
    utils.make_dir(out_path)
    for pair_i in feat_pairs:
        dataset_i=concat_helper(pair_i)
        x_i,y_i=pair_i
        out_i=out_path+"/nn"+str(x_i)+"_"+str(y_i)
        dataset_i.save(out_i)

def make_concat_helper(args_dict):
    datasets=ensemble.data.get_datasets(**args_dict)[0]
    def concat_helper(pair):
        x,y=pair
        return basic.unify_datasets([datasets[x],datasets[y]])
    return concat_helper

def get_pairs(sep_matrix):
    diff_matrix=np.abs(sep_matrix-np.diag(sep_matrix))
    pair_cats=np.argmax(diff_matrix,axis=0)
    feat_pairs,ids=[],Set()
    for i,x_i in enumerate(pair_cats):
        pair_id=cantor_paring(i,x_i)
        if(not pair_id in ids):
            ids.add(pair_id)
            ids.add(cantor_paring(x_i,i))
            feat_pairs.append((i,x_i))
    return feat_pairs

def cantor_paring(k1,k2):
    return (k1+k2)*(k1+k2+1)/2 +k2
