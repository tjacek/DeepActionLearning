import numpy as np
import deep.reader,utils
import basic.extr,deep.preproc
import basic.group,preproc.sampling

def group_ts_extractor(in_path,nn_path,out_path):
    def extr_helper(in_path_i,out_path_i):
        nn_path_i=nn_path+'/'+in_path_i.split("/")[-1]
        ts_extractor=build_ts_extractor(nn_path_i)
        ts_extractor(in_path_i,out_path_i)
    grup_fun=basic.group.GroupFun(extr_helper)
    grup_fun(nn_path,out_path)

def group_extractor(in_path,nn_path,out_path):
    extractor_factory=FrameExtractorFactory()
    def extr_helper(model_path_i,seq_path_i):
        extr_i=extractor_factory(model_path_i)
        extr_i(in_path,seq_path_i)
    grup_fun=basic.group.GroupFun(extr_helper)
    grup_fun(nn_path,out_path)    

def build_ts_extractor(nn_path):
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    conv.preproc=deep.preproc.ts_preproc(preproc.sampling.SplineUpsampling())
    return basic.extr.TimeSeriesExtractor(conv,feat_fun=False)

class FrameExtractorFactory(object):
    def __init__(self,cat_feat=True,n_frames=4):
        self.cat_feat = cat_feat
        self.n_frames=n_frames
        
    def __call__(self,nn_path):
        nn_reader=deep.reader.NNReader()
        conv=nn_reader(nn_path)
        conv.preproc=deep.preproc.FramePreproc(self.n_frames)#deep.preproc.ts_preproc
        if(self.cat_feat):
            def conv_helper(action_i):
               dist_i=conv.get_distribution(action_i)
               return dist_i#[[feat_j] for feat_j in list(dist_i)]
        else:
            conv_helper=lambda action_i:conv(action_i.as_array())
        return basic.extr.FrameExtractor(conv_helper)