import numpy as np 
import utils

class ActionGroup(object):
    def __init__(self, actions):
        self.actions = actions
    
    def __len__(self):
        return len(self.actions)

    def __getitem__(self, key):
        return self.actions[key]
    
    def raw(self):
        if(type(self.actions)==dict):
            return self.actions.values()
        return self.actions

    def select(self,selector=None,as_group=False):
        if(not selector):
            selector=lambda action_i:(action_i.person % 2)==1
        train,test=utils.split(self.actions,selector)
        if(as_group):
            return ActionGroup(train),ActionGroup(test)
        return train,test
    
    def normalization(self):
        feats=self.as_array()
        mean_feats=np.mean(feats,axis=0)
        std_feats=np.std(feats,axis=0)
        for action_i in self.actions:
            img_seq_i=action_i.as_array()
            img_seq_i-=mean_feats
            img_seq_i/=std_feats
            action_i.img_seq=list(img_seq_i)

    def as_array(self):
        feats=[]
        for action_i in self.actions:
            feats+=action_i.img_seq
        return np.array(feats)

class Action(object):
    def __init__(self,img_seq,name,cat,person):
        self.img_seq=img_seq
        self.name=name
        self.cat=cat
        self.person=person
    
    def __str__(self):
        return self.name

    def __len__(self):
        return len(self.img_seq)
    
    def __call__(self,fun,whole_seq=True):
        print(self.name)
        if(whole_seq):
            new_seq=fun(self.img_seq)
        else:
            new_seq=[ fun(img_i) for img_i in self.img_seq]
        return Action(new_seq,self.name,self.cat,self.person)	
    
    def clone(self,img_seq):
        return Action(img_seq,self.name,self.cat,self.person)

    def dim(self):
        frame=self.img_seq[0]
        if(type(frame)==list):
            return len(frame)
        return frame.shape[0]

    def as_array(self):
        return np.array(self.img_seq)

    def as_features(self):
        action_array=self.as_array().T
        return [ feature_i for feature_i in action_array]

    def as_pairs(self):#,norm=255.0):
        #norm_imgs=[ (img_i/norm) 
        #            for img_i in self.img_seq]
        return [ (self.cat,img_i) for img_i in norm_imgs]

def by_cat(actions):
    cats=[action_i.cat for action_i in actions]
    actions_by_cat={ cat_i:[] for cat_i in np.unique(cats)}
    for action_i in actions:
        actions_by_cat[action_i.cat].append(action_i)
    return actions_by_cat

def person_rep(actions):
    reps={}
    for action_i in actions:
        action_id=str(action_i.cat)+str(action_i.person)
        if(not action_id in reps):
            reps[action_id]=action_i
    return reps.values()