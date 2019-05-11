import matplotlib.pyplot as plt
import ensemble,ensemble.pick

class SelectionExper(object):
    def __init__(self,clf,dict_arg,detector_path,metric=None):
        if(not metric):
            metric=ensemble.pick.mean_cat
        self.clf=clf
        self.quality_args={ 'dict_arg':dict_arg, 'detector_path':detector_path,
                            'quality_metric':metric}
    
    def __call__(self,ens_arg,n_datasets=20):
        x=range(n_datasets+1)[1:]
        y=[self.single_exp(ens_arg,x_i) for x_i in x ]
        plt.plot(x,y)
        plt.show() 
        return y

    def single_exp(self,ens_arg,x_i):
        ens_i=self.get_ensemble(x_i)
        return ens_i(show=False,**ens_arg)[0][0]

    def get_ensemble(self,n_cls):
        quality=ensemble.pick.clf_quality(**self.quality_args)
        allowed_list=ensemble.pick.selectors.sort_list(quality,n_cls)
        return ensemble.Ensemble(self.clf,selector=allowed_list)