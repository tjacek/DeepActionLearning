import numpy as np
import deep.reader,deep.preproc,deep.extr
import utils,basic.extr

class PersonFeatures(object):
    def __init__(self,conv_nns):
        self.conv_nns=conv_nns

    def __call__(self,action_i):
        all_dist=np.array([conv_j.get_distribution(action_i)
                            for conv_j in self.conv_nns])
        all_dist=all_dist[:,:,0]
        return np.array(all_dist).T

def build_person_features(in_path,nn_path,out_path,n_frames=4):
    factory=deep.extr.FrameExtractorFactory(cat_feat=True)
    frame_extractor=factory(nn_path)
    frame_extractor(in_path,out_path)