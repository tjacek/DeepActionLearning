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
        new_sizes=[2*self.seq_size,self.seq_size/2]
        sides=[True,False]
        agum_seqs=[]
        for size_j in new_sizes: 
            for side_i in sides:
            	agum_seqs.append(warp_seq(feat_i,size_j,self.seq_size,side_i))
        agum_seqs=[interpolate(self.max_size,agum_i) for agum_i in agum_seqs]
        return agum_seqs

def warp_seq(feat_i,new_size,warp_size,side=True):
    print(feat_i.shape)
    start,end=feat_i[:warp_size],feat_i[warp_size:]
    if(side):        
        start=interpolate(new_size,start)
    else: 
        end=interpolate(new_size,end)
    return np.concatenate([start,end])

def interpolate(new_size,feat_i):
    old_x=get_x(feat_i.shape[0])
    cs=CubicSpline(old_x,feat_i)
    new_x=get_x(new_size)
    return cs(new_x)

def get_x(n):
    x=np.arange(n)
    x=x.astype(float)
    x/=float(n)
    return x

agum=Agumentation(SamplingAgum())
agum('datasets2/up_full','datasets2/agum')