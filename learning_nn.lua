--require 'spatial_nn/simple_convnet'
require 'volumetric_nn/vol_convnet'

function learning(in_path,out_path)
  hyper_params=default_hyper_params()
  input_files=get_input_files(path)
  dataset=get_dataset(input_files)
  local model=create_model()
  global_vars(model)
  local epoch=1
  local suboptimal_model=true
  while suboptimal_model do
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Epoch ")
    print(epoch)
    total_valid=train(model,dataset.train,dataset.train_labels,hyper_params)
    print(total_valid)
    suboptimal_model= total_valid < 0.95
    test(model,dataset.test,dataset.test_labels,hyper_params)
    epoch=epoch+1
  end
  torch.save(out_path,model)
end

function get_input_files(path)
  local input_files={}
  input_files.train=path .. "train.tensor"
  input_files.train_labels=path .. "train_labels.tensor"
  input_files.test=path .. "test.tensor"
  input_files.test_labels=path .. "test_labels.tensor"
  return input_files
end

function get_dataset(input_files)
  local dataset={}
  for name, file in pairs(input_files) do
    dataset[name]=torch.load(file)
  end
  return dataset
end

--path="/home/user/Desktop/dataset_3/" 
--learning(path,"nn_vol")

