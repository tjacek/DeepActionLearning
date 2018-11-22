import numpy as np
#numpy.random
from sklearn.mixture import GaussianMixture
import basic.extr

def gausian_feats(ts,max_comp=5):
#    feat_i=np.expand_dims(feat_i,axis=0).T
    print(ts.shape)

    mixtures,bic_values=zip(*[fit_gaussian(ts,i+1) for i in range(max_comp)])
    n_comp=np.argmin(bic_values)
    bic_n=bic_values[n_comp]
    print("n_components:%d bic:%.2f" % (n_comp,bic_n))
    feats=[n_comp,bic_n]
    return feats+prepare_feats(mixtures[-1])

def fit_gaussian(ts,n):
    mixture=GaussianMixture(n_components=n,
                   covariance_type="full", max_iter=20, random_state=0)
    samples=rejection_sampling(ts,size=500)
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
    means=np.squeeze(mixture.means_) 
    covars=np.squeeze(mixture.covariances_)
    indexes=np.argsort(means,axis=0)
    means=[ means[i] for i in indexes]
    covars=[ covars[i] for i in covars]
    return means + covars

extract=basic.extr.Extractor(gausian_feats)

extract( "../_LSTM/all", "gauss.txt")
