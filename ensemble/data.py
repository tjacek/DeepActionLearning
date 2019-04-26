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
        if(type(deep_path)==str):
            deep_paths=utils.bottom_files(deep_path)
        else:
            deep_paths=deep_path
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
    n_feats=datasets[0].dim()        
    print("Number of feats %d " % n_feats)
    return datasets,n_feats

def read_data(in_path,norm=True):
    dataset=basic.read_dataset(in_path)
    if(norm):
        dataset.norm()
    return dataset

def to_ensemble_samples(datasets,split=True):
    ens_insts=[ dataset_i.to_instances() for dataset_i in datasets]
    def ens_helper(names):
        return { name_j:[insts_i[name_j].data.reshape(1, -1) 
                    for insts_i in ens_insts]
                        for name_j in names} 
    if(split):
        train,test=datasets[0].to_instances().split()
        return {'train':ens_helper(train.names()),
                'test':ens_helper(test.names())}
    else:
        return ens_helper(datasets[0].names)

def to_train_ensemble(datasets):
    return [data_i.split()[0].X for data_i in datasets]

def feat_selection(handcrafted_path,deep_path,out_path,feats=100):
    utils.make_dir(out_path)
    datasets,n_feats=get_datasets(handcrafted_path,deep_path,feats)
    paths=utils.bottom_files(deep_path)
    all_out_paths=utils.switch_paths(out_path,paths)
    for out_i,dataset_i in zip(all_out_paths,datasets):
        print(out_i)
        dataset_i.save(out_i)