import numpy as np

def z_norm(feat_i):
    mean_i= np.mean(feat_i)
    std_i=  np.std(feat_i)
    feat_i-=mean_i
    feat_i/=std_i
    return feat_i