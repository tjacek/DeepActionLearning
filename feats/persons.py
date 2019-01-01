import numpy as np
import deep.reader,deep.preproc
import utils,basic.extr

class PersonFeatures(object):
    def __init__(self,conv_nns):
        self.conv_nns=conv_nns

    def __call__(action_i):
        all_dist=np.array([conv_j.get_distribution(action_i)
                            for conv_j in self.conv_nns])
        all_dist=all_dist[:,:,0]
        return np.array(all_dist).T

def build_person_features(in_path,n_frames=4):
    nn_paths=utils.bottom_files(in_path)
    nn_reader=deep.reader.NNReader()
    preproc=deep.preproc.FramePreproc(n_frames)
    conv_nns=[nn_reader(nn_path_i) for nn_path_i in nn_paths]
    for conv_i in conv_nns:
    	conv_i.preproc=preproc
    return basic.extr.FrameExtractor( PersonFeatures(conv_nns))