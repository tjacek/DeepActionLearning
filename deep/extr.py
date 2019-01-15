import numpy as np
import deep.reader,utils
import basic.extr,deep.preproc

def ensemble_extractor(in_path,nn_path,out_path):
    model_paths=utils.bottom_dirs(in_path)
    seqs_paths=utils.switch_paths(out_path,model_paths)
    utils.make_dir(out_path)
    for model_path_i,seq_path_i in zip(model_paths,seqs_paths):
        extr_i=build_frame_extractor(model_path_i,cat_feat=True,n_frames=4)
        extr_i(in_path,out_path_i)

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