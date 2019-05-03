import numpy as np,scipy.stats
import seq.io,basic.group,basic.extr,utils


def group_btf(in_path,out_path):
    utils.make_dir(out_path)
    ts_extr=basic.extr.TimeSeriesExtractor(basic_stats, feat_fun=True)  
    grup_fun=basic.group.GroupFun(ts_extr,dirs=True)
    grup_fun(in_path,out_path)    

def basic_stats(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0]
    return [np.mean(feat_i),np.std(feat_i),scipy.stats.skew(feat_i)]

def extended_stats(feat_i):
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0,0.0]
    return [np.mean(feat_i),np.std(feat_i),scipy.stats.skew(feat_i),time_corl(feat_i)]

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    #step=1.0/float(n_size)
    x_i=np.arange(float(n_size),step=1.0)#1.0,step=step)
    #x_i=x_i.astype(float)
    #x_i/=float(n_size)
    return scipy.stats.pearsonr(x_i,feat_i)[0]