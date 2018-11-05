import ensemble.tools,basic,utils

def select_feats(in_path,out_path,n_feats=100):
    deep_paths=utils.bottom_files(in_path)
    deep_datasets=[ basic.read_dataset(path_i) for path_i in deep_paths]
    deep_datasets=[ensemble.tools.rfe_selection(deep_i,n=n_feats) 
                        for deep_i in deep_datasets]
    utils.make_dir(out_path)
    for i,data_i in enumerate(deep_datasets):
        out_i=  out_path+"/"+deep_paths[i].split("/")[-1]
        data_i.save(out_i)	