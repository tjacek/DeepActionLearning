import numpy as np
#numpy.random
from sklearn.mixture import GaussianMixture
import basic.extr

def gausian_feats(ts,max_comp=5):
    print(ts.shape)
    mixtures,bic_values=zip(*[fit_gaussian(ts,i+1) for i in range(max_comp)])
    n_comp=np.argmin(bic_values)
    bic_n=bic_values[n_comp]
    print("n_components:%d bic:%.2f" % (n_comp,bic_n))
    feats=[n_comp,bic_n]
    return feats+prepare_feats(mixtures[-1])

def fixed_gaussian(ts_i,n_comp=5):
    if(n_comp==0):
        return [0,0]
    mixture_i,bic_i=fit_gaussian(ts_i,n_comp)
    feats=[n_comp,bic_i]
    return feats+prepare_feats(mixture_i)

def fit_gaussian(ts,n):
    mixture=GaussianMixture(n_components=n,
                   covariance_type="full", max_iter=20, random_state=0)
    samples=rejection_sampling(ts,size=1000)
    samples=np.expand_dims(samples,axis=0).T
    mixture.fit(samples)
    return mixture,mixture.bic(samples)

def rejection_sampling(feats,size=10000):
    seq_len=feats.shape[0]
    feats-=np.min(feats)
    feats/=np.max(feats)
    samples=np.arange(seq_len)/float(seq_len-1)
    indexes=np.random.randint(seq_len,size=size)
    threshold=np.random.uniform( size=size)
    return np.array([ samples[i] for i in indexes
                        if( feats[i]>threshold[i])])

def prepare_feats(mixture):
    if(mixture.n_components>1):
        means,covars=np.squeeze(mixture.means_),np.squeeze(mixture.covariances_)
    else:
        means,covars=mixture.means_,mixture.covariances_
    indexes=np.argsort(means,axis=0)
    means=[ means[i] for i in indexes]
    covars=[ covars[i] for i in indexes]    
    return means + covars