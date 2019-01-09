import numpy as np
import deep.reader
import basic.extr,deep.preproc

def build_ts_extractor(nn_path):
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    conv.preproc=deep.preproc.ts_preproc
    return basic.extr.TimeSeriesExtractor(conv,feat_fun=False)

def build_frame_extractor(nn_path,cat_feat=False,n_frames=4):
    nn_reader=deep.reader.NNReader()
    conv=nn_reader(nn_path)
    conv.preproc=FramePreproc(n_frames)#deep.preproc.ts_preproc
    if(cat_feat):
        def conv_helper(action_i):
            dist_i=conv.get_distribution(action_i)
            return [[feat_j] for feat_j in list(dist_i)]
    else:
        conv_helper=lambda action_i:conv(action_i.as_array())
    return basic.extr.FrameExtractor(conv_helper)