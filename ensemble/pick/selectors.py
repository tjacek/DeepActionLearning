import numpy as np 

class CatSelector(object):
    def __init__(self, allowed_set):
        allowed_set=Set(allowed_set)
        self.selector = lambda inst_i: (inst_i.cat in allowed_set)

    def __call__(self,datasets):
        datasets=[ data_i.split(self.selector)[0] for data_i in datasets]
        return [ data_i.integer_labels() for data_i in datasets]

class DatasetSelector(object):
    def __init__(self,allowed_list):
        self.allowed_list=allowed_list

    def __call__(self,datasets):
        return [datasets[i] for i in self.allowed_list]

def make_data_selector(dict_arg,detector_path,
                        clf_critterion=None,selection_type=15):
    if(not clf_critterion):
        clf_critterion=diagonal_criterion
    inliners_matrix=feats_inliners(dict_arg,detector_path)
    raw_quality=clf_critterion(inliners_matrix)
    if(type(selection_type)==int):
        n_cls= raw_quality.shape[0] - selection_type
        allowed_list=np.argsort(raw_quality)[n_cls:]
    else:
        allowed_list=[ i for i,bool_i in enumerate(raw_quality)
                        if(bool_i==1.0)]
    return DatasetSelector(allowed_list)
