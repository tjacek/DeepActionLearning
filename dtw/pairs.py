import numpy as np
import utils

class DTWPairs(object):
    def __init__(self,pairs):
        self.pairs=pairs
    
    def __getitem__(self, key):
        return self.pairs[key]

    def get_descs(self):
        return dataset.instances.get_descs(self.pairs.keys())

    def get_vector(self,name_i,names):
        return [ self.pairs[name_i][name_j] for name_j in names]
 
    def as_matrix(self):
        insts=self.get_descs() 
        names=insts.names()
        distance=[ self.get_vector(name_i,names) 
                    for name_i in names]
        X=np.array(distance)
        y,persons=insts.cats(),insts.persons()
        return X,y,persons

    def as_instances(self):
        insts=self.get_descs()
        train,test=insts.split()
        train_names= train.names()
        def feat_helper(inst_i):
            return self.get_vector(inst_i.name,train_names)
        for inst_i in insts.raw():
            inst_i.data=feat_helper(inst_i)
        return insts

    def save(self,out_path):
        dtw_tuples=as_tuples(self.pairs)
        text_pairs=[",".join(tuple_i) for tuple_i in dtw_tuples]
        utils.save_string(out_path,text_pairs)

def make_pairwise_distance(actions,dtw_metric):
    pairs_dict={ name_i:{name_i:0.0}
                    for name_i in actions.keys()}
    names=list(actions.keys())
    n_names=len(actions.keys())
    for i in range(1,n_names):
        print(i)
        for j in range(0,i):    
            action_i=actions[names[i]]#pair_i[0]]
            action_j=actions[names[j]]#pair_i[1]]
            distance=dtw_metric(action_i.as_array(),action_j.as_array())
            pairs_dict[action_i.name][action_j.name]=distance
            pairs_dict[action_j.name][action_i.name]=distance
    return DTWPairs(pairs_dict)

def as_tuples(dtw_distance):
    names=dtw_distance.keys()
    return [ (name_i,name_j,str(dtw_distance[name_i][name_j]))
                for name_i in names
                    for name_j in names]