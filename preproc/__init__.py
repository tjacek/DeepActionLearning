import numpy as np
import seq.io

class Preproc(object):
    def __init__(self,preproc_fun=None):
        if(not preproc_fun):
            preproc_fun=z_norm
        self.preproc_fun=preproc_fun
        
    def __call__(self,in_path,out_path):
    	read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False)
        actions=read_actions(in_path)
        def action_helper(action_i):
            feats=[ self.preproc_fun(feat_i) 
                        for feat_i in action_i.as_features()]
            img_seq=np.array(feats).T
            return action_i.clone(img_seq)
        actions=[action_helper(action_i) 
                        for action_i in actions]
        save_actions=seq.io.ActionWriter(False)
        save_actions(actions,out_path)

class Pipeline(object):
    def __init__(self,all_fun):
        self.all_fun=all_fun

    def __call__(self,feat_i):
        for fun_i in self.all_fun:
            feat_i= fun_i(feat_i)
        return feat_i