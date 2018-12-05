import numpy as np
import seq.io
from scipy.interpolate import CubicSpline

class Agumentation(object):
    def __init__(self,agum_fun):
        self.agum_fun=agum_fun

    def __call__(self,in_path,out_path):
        read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False,as_group=True)
        actions=read_actions(in_path)
        train,test=actions.select()
        agum_actions=[]
        for action_i in train:
            agum_actions+=self.agum_action(action_i)
        save_actions=seq.io.ActionWriter(False)
        save_actions(agum_actions,out_path)	
    
    def agum_action(self,action_i):
    	print(action_i.name)
        feats=action_i.as_features()
        agum_feats=[self.agum_fun(feat_i) 
                        for feat_i in feats]
        agum_feats=map(list,zip(*agum_feats)) 
        def action_helper(j,feats_j):
            img_seq=np.array(feats_j).T.tolist()
            action_ij=action_i.clone(img_seq) #seq.Action(img_seq,agum_name, action_i.cat,action_i.person) #
            action_ij.name=action_i.name+str(j)
            return action_ij
        agum_actions=[ action_helper(j,feats_j) 
                        for j,feats_j in enumerate(agum_feats)]
        print(action_i.name)
        return agum_actions

class SamplingAgum(object):
    def __init__(self,max_size=128,seq_size=16):
        self.max_size=max_size
        self.seq_size=seq_size

    def __call__(self,feat_i):
        #start,end=feat_i[:self.seq_size],feat_i[self.seq_size:]
        #start_i=CubicSpline(self.seq_size*2,start)
        #agum=np.concatenate([start_i,end])
        #agum=CubicSpline(self.max_size,agum)
        return [feat_i]

agum=Agumentation(SamplingAgum())
agum('datasets2/up_full','datasets2/agum')