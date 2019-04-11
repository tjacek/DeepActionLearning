import numpy as np
import cv2
from scipy.interpolate import CubicSpline
import seq.io,seq.tools,utils

class SplineUpsampling(object):
    def __init__(self,new_size=128):
        self.new_size=new_size

    def __call__(self,feat_i):
        old_size=feat_i.shape[0]
        old_x=np.arange(old_size)
        old_x=old_x.astype(float)  
        step=float(self.new_size)/float(old_size)
        old_x*=step  	
        cs=CubicSpline(old_x,feat_i)
        new_x=np.arange(self.new_size)
        print(new_x.shape)
        return cs(new_x)

def as_imgs(in_path,out_path):
    read_actions=seq.io.build_action_reader(img_seq=False,as_dict=False)
    actions=read_actions(in_path)
    utils.make_dir(out_path)
    for action_i in actions:
        name_i=action_i.name
        img_i=action_i.as_array()
        img_i*=10.0
        out_i=out_path+'/'+name_i+".png"
        cv2.imwrite(out_i,img_i)
  	
#seq.io.normalize("../wrap/mra_preproc/raw/all","../wrap/mra_preproc/norm/all")
#spline_upsampling=basic.preproc.Preproc(SplineUpsampling())
#spline_upsampling("../wrap/mra_preproc/norm/max_z","../wrap/mra_preproc/up/max_z")
#seq.tools.concat("../wrap/mra_preproc/up/all","../wrap/mra_preproc/up/max_z/", "../wrap/mra_preproc/full")
#as_imgs('../series/up_norm_all','../series/img_actions')