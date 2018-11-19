import seq.io
import instances

class Extractor(object):
    def __init__(self,get_feats):
        self.get_feats=get_feats

    def __call__(self,in_path,out_path):
        read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False)
        actions=read_actions(in_path)
        def action_helper(action_i):
            cat,person,name=action_i.cat,action_i.person,action_i.name
            data=self.get_feats(action_i)
            return instances.Instance(data,cat,person,name)
        insts=instances.InstsGroup([ action_helper(action_i) 
        	                            for action_i in actions])
        insts.to_txt(out_path)
    