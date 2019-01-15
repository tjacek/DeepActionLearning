import numpy as np,scipy.stats
import seq.io,basic.group

def group_btf(in_path,out_path):
    ts_extr=TimeSeriesExtractor(basic_stats, feat_fun=True)  
    grup_fun=basic.group.GroupFun(ts_extr)
    grup_fun(in_path,out_path)    

def basic_stats(feat_i):
    return [np.mean(feat_i),np.std(feat_i),scipy.stats.skew(feat_i)]