import numpy as np
import seq.io,seq.tools

class Preproc(object):
    def __init__(self,preproc_fun=None):
        self.preproc_fun=preproc_fun
            
    def __call__(self,actions):
        def action_helper(action_i):
            feats=[ self.preproc_fun(feat_i) 
                        for feat_i in action_i.as_features()]
            img_seq=np.array(feats).T
            return action_i.clone(img_seq)
        if(type(actions)==dict):
            return { action_i.name:action_helper(action_i) 
                        for action_i in actions.values()}
        return  [action_helper(action_i) 
                    for action_i in actions]

    def save(self,in_path,out_path):
        read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False)
        actions=read_actions(in_path)
        actions=self.transform(actions)
        save_actions=seq.io.ActionWriter(False)
        save_actions(actions,out_path)

class Pipeline(object):
    def __init__(self,all_fun):
        self.all_fun=all_fun

    def __call__(self,feat_i):
        for fun_i in self.all_fun:
            feat_i= fun_i(feat_i)
        return feat_i