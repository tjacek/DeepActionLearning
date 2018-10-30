import basic,utils

def get_datasets(handcrafted_path=None,deep_path=None):
    if(not handcrafted_path and not deep_path):
        raise Exception("No dataset paths")
    handcrafted_dataset,deep_datasets=None,None
    if(handcrafted_path):
        handcrafted_dataset=basic.read_dataset(handcrafted_path)
    if(deep_path):
        deep_paths=utils.bottom_files(deep_path)
        if(len(deep_paths)==0):
            raise Exception("No datasets at " + deep_paths)
        deep_datasets=[ basic.read_dataset(path_i) for path_i in deep_paths]
	return {"handcrafted":handcrafted_dataset,"deep":deep_datasets}