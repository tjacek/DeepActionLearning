import numpy as np
import deep.reader,utils
import basic.extr,deep.preproc

class EnsembleExtr(object):
    def __init__(self,fun):
        self.fun=fun

    def __call__(in_path,out_path):
        all_paths=utils.bottom_dirs(in_path)
        new_paths=utils.switch_paths(out_path,all_paths)
        for in_i,out_i in zip(all_paths,new_paths):
            self.fun(in_i,out_i):

def ensemble_frame_extractor(nn_path,cat_feat=True,n_frames=4):
    return EnsembleExtr(build_frame_extractor(nn_path,cat_feat,n_frames))

def build_ts_extractor(nn_path):
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    conv.preproc=deep.preproc.ts_preproc
    return basic.extr.TimeSeriesExtractor(conv,feat_fun=False)

def build_frame_extractor(nn_path,cat_feat=True,n_frames=4):
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    conv.preproc=deep.preproc.FramePreproc(n_frames)#deep.preproc.ts_preproc
    if(cat_feat):
        def conv_helper(action_i):
            dist_i=conv.get_distribution(action_i)
            return dist_i#[[feat_j] for feat_j in list(dist_i)]
    else:
        conv_helper=lambda action_i:conv(action_i.as_array())
    return basic.extr.FrameExtractor(conv_helper)