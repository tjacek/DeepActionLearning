import numpy as np
import seq.io, basic,basic.instances
import gauss,gauss.tools

def optim_gaussian(in_path):    
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_actions(in_path)
    fourier_smooth=gauss.tools.FourierSmooth()
    def action_helper(action_i):
        features=[fourier_smooth(feat_i) 
                    for feat_i in action_i.as_features()]
        optim=[gauss.tools.count_maxs(feat_i) 
                    for feat_i in features]
        return basic.instances.make_instance(optim,action_i)
    insts=[ action_helper(action_i) for action_i in actions]
    dataset=basic.to_dataset(insts)
    n_feats=np.amax(dataset.X,axis=0)
    print(n_feats)


def optim_gauss(ts_i):
    fourier_smooth=gauss.tools.FourierSmooth()
    ts_i=fourier_smooth(ts_i)
    n_optims=gauss.tools.count_maxs(ts_i)
    return gauss.fixed_gaussian(ts_i,n_comp=n_optims)