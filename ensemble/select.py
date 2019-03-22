import ensemble.data,ensemble.tools,basic,utils
import ensemble.outliner

def select_feats(dict_arg,detector_path,cat_j=0):
    datasets,n_feats=ensemble.data.get_datasets(dict_arg,None,None)
    detectors=ensemble.outliner.read_detectors(detector_path)
    quality=[ detectors.cat_separation(i,dataset_i,cat_j)
                for i,dataset_i in enumerate(datasets)]
    print(quality)
#def select_feats(in_path,out_path,n_feats=100):
#    deep_paths=utils.bottom_files(in_path)
#    deep_datasets=[ basic.read_dataset(path_i) for path_i in deep_paths]
#    for deep_i in deep_datasets:
#    	deep_i.norm()
#    deep_datasets=[ensemble.tools.rfe_selection(deep_i,n=n_feats) 
#                        for deep_i in deep_datasets]
#    utils.make_dir(out_path)
#    for i,data_i in enumerate(deep_datasets):
#        out_i=  out_path+"/"+deep_paths[i].split("/")[-1]
#        data_i.save(out_i)	