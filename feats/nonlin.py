import numpy as np 
import basic.extr,gauss.tools

def non_linear(feat_i):
    feat_i=gauss.tools.FourierSmooth()(feat_i)
    feat_i-= np.amin(feat_i)
    feat_i/= np.amax(feat_i)
    feat_i+=0.01
    filtr=np.array([-0.25,-0.25,1.0,-0.25,-0.25])
    resid_i=np.convolve(feat_i,filtr,mode="valid")
    resized_feat_i=feat_i[2:]
    resized_feat_i=resized_feat_i[:-2]
    nonlinearity=np.abs(resid_i/resized_feat_i) 
    return [np.mean(nonlinearity),np.median(nonlinearity),np.amax(nonlinearity)]
