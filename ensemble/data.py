import basic,utils
import ensemble.tools

def get_datasets(handcrafted_path,deep_path,feats):
    datasets_dict=read_datasets(handcrafted_path,deep_path)
    feat_reduction(datasets_dict,hc_feats=feats[0],deep_feats=feats[1])
    return preproc_dataset(datasets_dict)

def read_datasets(handcrafted_path=None,deep_path=None):
    if(not handcrafted_path and not deep_path):
        raise Exception("No dataset paths")
    handcrafted_dataset,deep_datasets=None,None
    if(handcrafted_path):
        handcrafted_dataset= read_data(handcrafted_path)
    if(deep_path):
        deep_paths=utils.bottom_files(deep_path)
        if(len(deep_paths)==0):
            raise Exception("No datasets at " + deep_path)
        deep_datasets=[read_data(path_i) for path_i in deep_paths]
    return {"handcrafted":handcrafted_dataset,"deep":deep_datasets}

def feat_reduction(datasets_dict,hc_feats=250,deep_feats=100):
    hc_data=datasets_dict['handcrafted']
    if(hc_data):
        datasets_dict['handcrafted']=ensemble.tools.rfe_selection(hc_data,hc_feats)
    if(datasets_dict['deep']):
        datasets_dict['deep']=[ ensemble.tools.rfe_selection(deep_i,deep_feats)  
                                    for deep_i in datasets_dict['deep']]
    return datasets_dict

def preproc_dataset(datasets_dict):
    datasets=datasets_dict["deep"]
    if(datasets_dict["handcrafted"]):
        hc_data=datasets_dict["handcrafted"]
        if(datasets_dict["deep"]):
            datasets=[basic.unify_datasets([hc_data,deep_i])
                        for deep_i in datasets]
        else:
            datasets=[hc_data]
    print("Number of feats %d " % datasets[0].dim())
    return datasets

def read_data(in_path,norm=True):
    dataset=basic.read_dataset(in_path)
    if(norm):
        dataset.norm()
    return dataset