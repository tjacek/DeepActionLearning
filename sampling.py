import numpy as np
from scipy.interpolate import CubicSpline
import basic.preproc

class SplineUpsampling(object):
    def __init__(self,new_size=101):
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

spline_upsampling=basic.preproc.Preproc(SplineUpsampling())
spline_upsampling('../series/all','../series/up_all')