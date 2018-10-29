import basic

def get_datasets(handcrafted_path=None,deep_path=None):
	if(not handcrafted_path and not deep_path):
		raise Exception("No dataset paths")
	handcrafted_dataset,deep_datasets=None,None
	if(handcrafted_path):
		handcrafted_dataset=basic.read_dataset(handcrafted_path)
	return {"handcrafted":handcrafted_dataset,"deep":deep_datasets}