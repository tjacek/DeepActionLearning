import numpy as np
import plot

class ClfStats(object):
    def __init__(self, clf_quality=None,inspect_acc=None):
        if(not clf_quality):
            clf_quality=diagonal_criterion
        self.clf_quality = clf_quality
        if(not inspect_acc):
            inspect_acc=correlation_acc
        self.inspect_acc=inspect_acc

    def __call__(self,clf_acc,dict_arg,detector_path):
        inliners_matrix=feats_inliners(dict_arg,detector_path)
        feats_quality=self.clf_quality(inliners_matrix)
        return self.inspect_acc(clf_acc,feats_quality)

def correlation_acc(clf_acc,feats_quality):
    X=np.stack([clf_acc,feats_quality])
    return np.corrcoef(X)[0][1]

def resiudals(clf_acc,feats_quality):
    #regr=utils.linear_reg(clf_acc,feats_quality)
    regr,pred_acc=plot.show_regres(feats_quality,clf_acc)
    #pred_acc=regr.predict(feats_quality)
    return np.mean(np.abs( clf_acc-pred_acc))